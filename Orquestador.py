import logging
import re
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd
import requests

ANYTHINGLLM_URL = "http://localhost:3001/api/v1"
API_KEY = "J42JHTB-5XFMA9H-J8Q46QH-NHV1DDH"
WORKSPACE_SLUG = "gemelo-digital-cognitivo-estanque"
REQUEST_TIMEOUT = 600
RESULTADOS_DIR = Path("resultados")

COLUMNAS_REQUERIDAS = [
    "tiempo_s",
    "nivel_m",
    "setpoint_m",
    "error_m",
    "Qin_m3s",
    "Qout_m3s",
    "control_pct",
]

PARAMETROS_PLANTA = {
    "A_m2": 0.02,
    "Cd": 0.6,
    "a_m2": 2e-4,
    "g_m_s2": 9.81,
    "h_max_m": 0.50,
    "Qin_max_m3s": 3e-4,
    "beta_m_5_2_s": 5.31e-4,
    "tau_s": 37.68,
}

MODO_SALIDA = {"1": "operador", "2": "tecnico"}
MODO_ANALISIS = {"1": "puntual", "2": "periodo", "3": "resumen", "4": "comparar", "5": "todos", "0": "salir"}
TIEMPO_IGNORAR_ARRANQUE_S = 20.0

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("orquestador_cdt")


@dataclass
class ResultadoLLM:
    exito: bool
    respuesta: str
    duracion_s: float
    error: str = ""


class OrquestadorCDT:
    def __init__(self, api_key: str, url: str, workspace_slug: str) -> None:
        self.api_key = api_key
        self.url = url.rstrip("/")
        self.workspace_slug = workspace_slug
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
        )
        RESULTADOS_DIR.mkdir(exist_ok=True)

    def verificar_conexion(self) -> bool:
        logger.info("Verificando conexion con AnythingLLM...")
        try:
            resp = self.session.get(f"{self.url}/workspaces", timeout=15)
            resp.raise_for_status()
            data = resp.json()
            encontrado = self._workspace_existe(data)
            if encontrado:
                logger.info("Conexion exitosa con AnythingLLM y workspace encontrado")
            else:
                logger.warning("Conexion con AnythingLLM OK, pero no se pudo confirmar el workspace '%s'", self.workspace_slug)
            return True
        except requests.HTTPError as exc:
            logger.error("Error HTTP al verificar conexion: %s", exc)
            return False
        except requests.RequestException as exc:
            logger.error("No fue posible conectar con AnythingLLM: %s", exc)
            return False

    def _workspace_existe(self, data: Any) -> bool:
        if isinstance(data, dict):
            candidatos = []
            for clave in ("workspaces", "data", "results"):
                valor = data.get(clave)
                if isinstance(valor, list):
                    candidatos.extend(valor)
            if not candidatos and "slug" in data:
                candidatos.append(data)
        elif isinstance(data, list):
            candidatos = data
        else:
            return False

        for item in candidatos:
            if isinstance(item, dict) and item.get("slug") == self.workspace_slug:
                return True
        return False

    def cargar_csv(self, ruta_csv: str | Path) -> pd.DataFrame:
        ruta = Path(ruta_csv)
        if not ruta.exists():
            raise FileNotFoundError(f"No existe el archivo CSV: {ruta}")

        logger.info("Cargando CSV: %s", ruta.name)
        df = pd.read_csv(ruta, encoding="utf-8")

        faltantes = [c for c in COLUMNAS_REQUERIDAS if c not in df.columns]
        if faltantes:
            raise ValueError(f"Faltan columnas requeridas: {', '.join(faltantes)}")

        df = df[COLUMNAS_REQUERIDAS].copy()

        for col in COLUMNAS_REQUERIDAS:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        n_nulos = int(df.isna().any(axis=1).sum())
        if n_nulos > 0:
            logger.warning("Se eliminaron %d filas con valores nulos", n_nulos)
            df = df.dropna().reset_index(drop=True)

        if df.empty:
            raise ValueError("El DataFrame quedo vacio despues de limpiar nulos")

        df = df.sort_values("tiempo_s").reset_index(drop=True)

        logger.info(
            "  Filas: %d | Duracion: %.1f s | Nivel: %.4f—%.4f m",
            len(df),
            float(df["tiempo_s"].iloc[-1] - df["tiempo_s"].iloc[0]),
            float(df["nivel_m"].min()),
            float(df["nivel_m"].max()),
        )
        return df

    def consultar_llm(self, prompt: str) -> ResultadoLLM:
        endpoint = f"{self.url}/workspace/{self.workspace_slug}/chat"
        payload = {"message": prompt, "mode": "query"}

        inicio = perf_counter()
        try:
            resp = self.session.post(endpoint, json=payload, timeout=REQUEST_TIMEOUT)
            duracion = perf_counter() - inicio
            resp.raise_for_status()
            data = resp.json()

            texto = self._extraer_texto_respuesta(data)
            texto = self.limpiar_respuesta_llm(texto)
            if not texto.strip():
                return ResultadoLLM(exito=False, respuesta="", duracion_s=duracion, error="Respuesta vacia del LLM")

            return ResultadoLLM(exito=True, respuesta=texto, duracion_s=duracion)
        except requests.Timeout:
            return ResultadoLLM(exito=False, respuesta="", duracion_s=perf_counter() - inicio, error=f"Timeout despues de {REQUEST_TIMEOUT} s")
        except requests.RequestException as exc:
            return ResultadoLLM(exito=False, respuesta="", duracion_s=perf_counter() - inicio, error=str(exc))

    @staticmethod
    def _extraer_texto_respuesta(data: dict[str, Any]) -> str:
        posibles = [
            data.get("textResponse"),
            data.get("response"),
            data.get("text"),
            (data.get("message") or {}).get("text") if isinstance(data.get("message"), dict) else None,
        ]
        for item in posibles:
            if isinstance(item, str) and item.strip():
                return item.strip()
        return ""

    def calcular_metricas(self, df: pd.DataFrame) -> dict[str, Any]:
        tiempo = df["tiempo_s"]
        nivel = df["nivel_m"]
        sp = df["setpoint_m"]
        error = df["error_m"]
        qin = df["Qin_m3s"]
        qout = df["Qout_m3s"]
        control = df["control_pct"]

        setpoint_inicial = float(sp.iloc[0])
        setpoint_final = float(sp.iloc[-1])
        delta_sp = setpoint_final - setpoint_inicial
        nivel_final = float(nivel.iloc[-1])

        overshoot_abs = max(0.0, float(nivel.max() - setpoint_final))
        overshoot_pct = ((overshoot_abs / abs(delta_sp)) * 100.0 if abs(delta_sp) > 1e-12 else (overshoot_abs / max(abs(setpoint_final), 1e-9)) * 100.0)

        banda_est = max(0.02 * max(abs(setpoint_final), 1e-9), 0.002)
        settling_time = self._calcular_settling_time(tiempo, nivel, setpoint_final, banda_est)

        metricas = {
            "muestras": int(len(df)),
            "t_inicio_s": float(tiempo.iloc[0]),
            "t_fin_s": float(tiempo.iloc[-1]),
            "duracion_s": float(tiempo.iloc[-1] - tiempo.iloc[0]),
            "dt_promedio_s": float(tiempo.diff().dropna().mean()) if len(df) > 1 else 0.0,
            "nivel_min_m": float(nivel.min()),
            "nivel_prom_m": float(nivel.mean()),
            "nivel_max_m": float(nivel.max()),
            "nivel_std_m": float(nivel.std(ddof=0)),
            "nivel_final_m": nivel_final,
            "sp_inicial_m": setpoint_inicial,
            "sp_final_m": setpoint_final,
            "error_abs_prom_m": float(error.abs().mean()),
            "error_max_abs_m": float(error.abs().max()),
            "error_final_m": float(error.iloc[-1]),
            "Qin_prom_m3s": float(qin.mean()),
            "Qout_prom_m3s": float(qout.mean()),
            "control_prom_pct": float(control.mean()),
            "control_max_pct": float(control.max()),
            "control_min_pct": float(control.min()),
            "control_std_pct": float(control.std(ddof=0)),
            "overshoot_abs_m": overshoot_abs,
            "overshoot_pct": float(overshoot_pct),
            "settling_time_s": settling_time,
            "estado_estable": bool(abs(float(error.iloc[-1])) <= banda_est),
        }

        metricas.update(self._metricas_zonas(nivel))
        metricas.update(self._metricas_control(control))
        return metricas

    @staticmethod
    def _calcular_settling_time(tiempo: pd.Series, nivel: pd.Series, sp_final: float, banda: float) -> float | None:
        dentro = (nivel - sp_final).abs() <= banda
        if not bool(dentro.any()):
            return None
        dentro_lista = dentro.to_list()
        t_lista = tiempo.to_list()
        for i in range(len(dentro_lista)):
            if all(dentro_lista[i:]):
                return float(t_lista[i])
        return None

    @staticmethod
    def _metricas_zonas(nivel: pd.Series) -> dict[str, float]:
        total = max(len(nivel), 1)
        normal = ((nivel >= 0.10) & (nivel <= 0.40)).sum() / total * 100.0
        prec = (((nivel >= 0.05) & (nivel < 0.10)).sum() + ((nivel > 0.40) & (nivel <= 0.45)).sum()) / total * 100.0
        alerta = ((nivel < 0.05) | (nivel > 0.45)).sum() / total * 100.0
        return {"pct_normal": float(normal), "pct_precaucion": float(prec), "pct_alerta": float(alerta)}

    @staticmethod
    def _metricas_control(control: pd.Series) -> dict[str, Any]:
        total = max(len(control), 1)
        pct_cerca_sat_alta = float((control >= 90).sum() / total * 100.0)
        pct_cerca_sat_baja = float((control <= 10).sum() / total * 100.0)
        rango = float(control.max() - control.min())
        es_ruidoso = bool(float(control.std(ddof=0)) > 10.0 or rango > 35.0)
        return {
            "control_rango_pct": rango,
            "pct_control_alto": pct_cerca_sat_alta,
            "pct_control_bajo": pct_cerca_sat_baja,
            "control_ruidoso": es_ruidoso,
        }

    def _detectar_eventos(self, df: pd.DataFrame) -> list[str]:
        eventos: list[str] = []
        sp = df["setpoint_m"]
        nivel = df["nivel_m"]
        control = df["control_pct"]
        tiempo = df["tiempo_s"]
        error = df["error_m"]

        df_post_arranque = df[df["tiempo_s"] >= TIEMPO_IGNORAR_ARRANQUE_S].copy()
        if df_post_arranque.empty:
            df_post_arranque = df.copy()

        tiempo_post = df_post_arranque["tiempo_s"]
        nivel_post = df_post_arranque["nivel_m"]
        control_post = df_post_arranque["control_pct"]

        cambios_sp = sp.diff().abs() > 1e-6
        if cambios_sp.any():
            idx = cambios_sp[cambios_sp].index[0]
            eventos.append(f"Cambio de setpoint detectado en t={tiempo.iloc[idx]:.1f} s: {sp.iloc[idx-1]:.3f} -> {sp.iloc[idx]:.3f} m")

        cruces_alta = nivel_post > 0.45
        if cruces_alta.any():
            idx = cruces_alta[cruces_alta].index[0]
            eventos.append(f"Nivel critico alto detectado en t={tiempo_post.iloc[idx]:.1f} s con h={nivel_post.iloc[idx]:.3f} m")

        cruces_baja = nivel_post < 0.05
        if cruces_baja.any():
            idx = cruces_baja[cruces_baja].index[0]
            eventos.append(f"Nivel critico bajo detectado en t={tiempo_post.iloc[idx]:.1f} s con h={nivel_post.iloc[idx]:.3f} m")

        if control_post.max() >= 99.9:
            idx = control_post.idxmax()
            eventos.append(f"Control cercano a saturacion alta en t={df.loc[idx, 'tiempo_s']:.1f} s con u={df.loc[idx, 'control_pct']:.1f}%")

        if control_post.min() <= 0.1:
            idx = control_post.idxmin()
            eventos.append(f"Control cercano a saturacion baja en t={df.loc[idx, 'tiempo_s']:.1f} s con u={df.loc[idx, 'control_pct']:.1f}%")

        if error.abs().mean() > 0.05:
            eventos.append("Error medio absoluto elevado respecto del setpoint")

        if len(df_post_arranque) > 10 and float(df_post_arranque["nivel_m"].std(ddof=0)) > 0.01:
            eventos.append("Fluctuacion visible del nivel en la ventana analizada")

        if len(df_post_arranque) > 10 and float(df_post_arranque["control_pct"].std(ddof=0)) > 10:
            eventos.append("Accion de control altamente variable")

        return eventos

    @staticmethod
    def limpiar_respuesta_llm(texto: str) -> str:
        if not texto:
            return ""
        texto = re.sub(r"<think>.*?</think>", "", texto, flags=re.DOTALL | re.IGNORECASE)
        texto = re.sub(r"```.*?```", "", texto, flags=re.DOTALL)
        return texto.strip()

    @staticmethod
    def extraer_estado(texto: str) -> str:
        texto_limpio = OrquestadorCDT.limpiar_respuesta_llm(texto)

        for linea in texto_limpio.splitlines():
            linea_limpia = linea.strip()
            if linea_limpia.upper().startswith("ESTADO:"):
                estado_txt = linea_limpia.split(":", 1)[1].strip().upper()
                if "ERROR_DE_DATOS" in estado_txt or "ERROR DE DATOS" in estado_txt:
                    return "ERROR_DE_DATOS"
                if "PRECAUCION" in estado_txt:
                    return "PRECAUCION"
                if "ALERTA" in estado_txt:
                    return "ALERTA"
                if "NORMAL" in estado_txt:
                    return "NORMAL"

        patrones = [r"\bERROR[_ ]?DE[_ ]?DATOS\b", r"\bALERTA\b", r"\bPRECAUCION\b", r"\bNORMAL\b"]
        texto_up = texto_limpio.upper()
        for patron in patrones:
            match = re.search(patron, texto_up)
            if match:
                return match.group(0).replace(" ", "_")
        return "NO_IDENTIFICADO"

    @staticmethod
    def _instrucciones_salida(estilo: str) -> str:
        if estilo == "operador":
            return (
                "MODO OPERADOR (OBLIGATORIO):\n"
                "- Responde en maximo 4 lineas.\n"
                "- Usa frases cortas, directas y faciles de entender.\n"
                "- No expliques teoria salvo que sea indispensable.\n"
                "- Prioriza: que pasa, por que importa y que hacer.\n"
                "- Usa exactamente este formato:\n"
                "Estado: [NORMAL / PRECAUCION / ALERTA / ERROR_DE_DATOS]\n"
                "Problema: ...\n"
                "Causa: ...\n"
                "Accion: ..."
            )

        return (
            "MODO TECNICO (OBLIGATORIO):\n"
            "- Responde en 1 o 2 parrafos breves.\n"
            "- Explica el comportamiento del nivel, el error, el tiempo de establecimiento y la calidad de la señal de control.\n"
            "- Si corresponde, menciona saturacion, ruido, sobretiro, variabilidad del control y posible causa raiz.\n"
            "- Usa lenguaje tecnico claro, pero no excesivamente largo.\n"
            "- Incluye una recomendacion final concreta.\n"
            "- Usa esta estructura:\n"
            "Estado: [NORMAL / PRECAUCION / ALERTA / ERROR_DE_DATOS]\n"
            "Analisis: ...\n"
            "Recomendacion: ..."
        )

    def _construir_prompt_puntual(self, nombre_csv: str, fila: pd.Series, tendencia_nivel: float, estilo: str) -> str:
        return f"""
Actua como capa cognitiva de un Gemelo Digital Cognitivo de un estanque de agua.
Debes responder SIEMPRE incluyendo una clasificacion explicita: NORMAL, PRECAUCION, ALERTA o ERROR_DE_DATOS.
{self._instrucciones_salida(estilo)}

Analiza este punto temporal del escenario {nombre_csv}.

Datos instantaneos:
- tiempo_s: {fila['tiempo_s']:.3f}
- nivel_m: {fila['nivel_m']:.5f}
- setpoint_m: {fila['setpoint_m']:.5f}
- error_m: {fila['error_m']:.5f}
- Qin_m3s: {fila['Qin_m3s']:.7f}
- Qout_m3s: {fila['Qout_m3s']:.7f}
- control_pct: {fila['control_pct']:.3f}
- tendencia_nivel_m_por_s: {tendencia_nivel:.6f}

Criterios orientativos:
- NORMAL si el nivel esta en rango operativo y la tendencia es coherente.
- PRECAUCION si hay acercamiento a umbrales, oscilacion o accion de control exigida.
- ALERTA si hay riesgo de desborde, vaciado critico, error sostenido alto o saturacion preocupante.
- ERROR_DE_DATOS si observas inconsistencia fisica o estructural.

Si el estilo es tecnico, usa la estructura Estado / Analisis / Recomendacion.
""".strip()

    def _construir_prompt_periodo(self, nombre_csv: str, desde: float, hasta: float, metricas: dict[str, Any], eventos: list[str], estilo: str) -> str:
        eventos_txt = "\n".join(f"- {e}" for e in eventos) if eventos else "- No se detectaron eventos llamativos"
        settling = f"{metricas['settling_time_s']:.3f}" if isinstance(metricas["settling_time_s"], (int, float)) and metricas["settling_time_s"] is not None else "No alcanzado"
        return f"""
Analiza el siguiente periodo temporal del escenario {nombre_csv} para un Gemelo Digital Cognitivo de un estanque de agua.
Responde obligatoriamente con una clasificacion: NORMAL, PRECAUCION, ALERTA o ERROR_DE_DATOS.
{self._instrucciones_salida(estilo)}

Periodo analizado: {desde:.1f} s a {hasta:.1f} s
Metricas agregadas:
- muestras: {metricas['muestras']}
- nivel_min_m: {metricas['nivel_min_m']:.5f}
- nivel_prom_m: {metricas['nivel_prom_m']:.5f}
- nivel_max_m: {metricas['nivel_max_m']:.5f}
- nivel_std_m: {metricas['nivel_std_m']:.6f}
- sp_final_m: {metricas['sp_final_m']:.5f}
- error_abs_prom_m: {metricas['error_abs_prom_m']:.5f}
- error_max_abs_m: {metricas['error_max_abs_m']:.5f}
- control_prom_pct: {metricas['control_prom_pct']:.3f}
- control_max_pct: {metricas['control_max_pct']:.3f}
- control_std_pct: {metricas['control_std_pct']:.3f}
- pct_control_alto: {metricas['pct_control_alto']:.2f}
- control_ruidoso: {metricas['control_ruidoso']}
- overshoot_pct: {metricas['overshoot_pct']:.2f}
- settling_time_s: {settling}
- pct_normal: {metricas['pct_normal']:.2f}
- pct_precaucion: {metricas['pct_precaucion']:.2f}
- pct_alerta: {metricas['pct_alerta']:.2f}

Eventos detectados:
{eventos_txt}

Si el estilo es tecnico, usa la estructura Estado / Analisis / Recomendacion.
""".strip()

    def _construir_prompt_resumen(self, nombre_csv: str, m_global: dict[str, Any], m_primera: dict[str, Any], m_segunda: dict[str, Any], eventos: list[str], estilo: str) -> str:
        eventos_txt = "\n".join(f"- {e}" for e in eventos) if eventos else "- No se detectaron eventos llamativos"
        settling = f"{m_global['settling_time_s']:.3f}" if isinstance(m_global["settling_time_s"], (int, float)) and m_global["settling_time_s"] is not None else "No alcanzado"
        return f"""
Actua como la capa cognitiva de un Gemelo Digital Cognitivo para control de nivel en un estanque de agua.
Debes emitir una clasificacion explicita: NORMAL, PRECAUCION, ALERTA o ERROR_DE_DATOS.
{self._instrucciones_salida(estilo)}

Analiza globalmente el escenario {nombre_csv}.

Metricas globales:
- duracion_s: {m_global['duracion_s']:.2f}
- muestras: {m_global['muestras']}
- nivel_min_m: {m_global['nivel_min_m']:.5f}
- nivel_prom_m: {m_global['nivel_prom_m']:.5f}
- nivel_max_m: {m_global['nivel_max_m']:.5f}
- nivel_final_m: {m_global['nivel_final_m']:.5f}
- sp_inicial_m: {m_global['sp_inicial_m']:.5f}
- sp_final_m: {m_global['sp_final_m']:.5f}
- error_abs_prom_m: {m_global['error_abs_prom_m']:.5f}
- error_max_abs_m: {m_global['error_max_abs_m']:.5f}
- error_final_m: {m_global['error_final_m']:.5f}
- Qin_prom_m3s: {m_global['Qin_prom_m3s']:.7f}
- Qout_prom_m3s: {m_global['Qout_prom_m3s']:.7f}
- control_prom_pct: {m_global['control_prom_pct']:.3f}
- control_max_pct: {m_global['control_max_pct']:.3f}
- control_min_pct: {m_global['control_min_pct']:.3f}
- control_std_pct: {m_global['control_std_pct']:.3f}
- pct_control_alto: {m_global['pct_control_alto']:.2f}
- pct_control_bajo: {m_global['pct_control_bajo']:.2f}
- control_ruidoso: {m_global['control_ruidoso']}
- overshoot_pct: {m_global['overshoot_pct']:.2f}
- settling_time_s: {settling}
- estado_estable: {m_global['estado_estable']}
- pct_normal: {m_global['pct_normal']:.2f}
- pct_precaucion: {m_global['pct_precaucion']:.2f}
- pct_alerta: {m_global['pct_alerta']:.2f}

Comparacion por mitades:
Primera mitad:
- nivel_prom_m: {m_primera['nivel_prom_m']:.5f}
- error_abs_prom_m: {m_primera['error_abs_prom_m']:.5f}
- control_prom_pct: {m_primera['control_prom_pct']:.3f}
- control_std_pct: {m_primera['control_std_pct']:.3f}

Segunda mitad:
- nivel_prom_m: {m_segunda['nivel_prom_m']:.5f}
- error_abs_prom_m: {m_segunda['error_abs_prom_m']:.5f}
- control_prom_pct: {m_segunda['control_prom_pct']:.3f}
- control_std_pct: {m_segunda['control_std_pct']:.3f}

Eventos detectados:
{eventos_txt}

Reglas importantes:
- Si el nivel esta estable pero la señal de control es muy ruidosa o muy cercana a saturacion, clasifica como PRECAUCION y recomienda revisar Kd o aplicar filtrado.
- NO clasifiques como PRECAUCION solo por el arranque inicial del escenario.
- En escenario de linea base, ignora como anomalia el nivel bajo y la saturacion que ocurren solo al inicio del llenado.
- Si en la segunda mitad del ensayo el sistema permanece estable, con error final bajo y sin saturacion persistente, clasifica como NORMAL.
- En modo tecnico, desarrolla el analisis del sistema usando las metricas disponibles.
- Comenta explicitamente:
  1) si el nivel converge o no al setpoint,
  2) si el error final es bajo o relevante,
  3) si el tiempo de establecimiento es aceptable,
  4) si la señal de control es suave, ruidosa o cercana a saturacion,
  5) si el comportamiento global del PID es adecuado.
- Si el sistema esta normal, no te limites a decir "todo bien"; explica brevemente por que esta normal.
""".strip()

    def _construir_prompt_comparacion(self, nombre_csv_1: str, nombre_csv_2: str, m1: dict[str, Any], m2: dict[str, Any], estilo: str) -> str:
        return f"""
Compara dos escenarios de un Gemelo Digital Cognitivo para control de nivel en estanque.
Entrega una clasificacion final para el escenario mas riesgoso y una conclusion comparativa.
{self._instrucciones_salida(estilo)}

Escenario A: {nombre_csv_1}
- nivel_prom_m: {m1['nivel_prom_m']:.5f}
- nivel_max_m: {m1['nivel_max_m']:.5f}
- error_abs_prom_m: {m1['error_abs_prom_m']:.5f}
- overshoot_pct: {m1['overshoot_pct']:.2f}
- control_prom_pct: {m1['control_prom_pct']:.3f}
- control_std_pct: {m1['control_std_pct']:.3f}
- pct_alerta: {m1['pct_alerta']:.2f}

Escenario B: {nombre_csv_2}
- nivel_prom_m: {m2['nivel_prom_m']:.5f}
- nivel_max_m: {m2['nivel_max_m']:.5f}
- error_abs_prom_m: {m2['error_abs_prom_m']:.5f}
- overshoot_pct: {m2['overshoot_pct']:.2f}
- control_prom_pct: {m2['control_prom_pct']:.3f}
- control_std_pct: {m2['control_std_pct']:.3f}
- pct_alerta: {m2['pct_alerta']:.2f}
""".strip()

    def _guardar_resultados(self, datos: list[dict[str, Any]] | dict[str, Any], nombre_archivo: str) -> str:
        ruta = RESULTADOS_DIR / nombre_archivo
        df = pd.DataFrame([datos]) if isinstance(datos, dict) else pd.DataFrame(datos)
        df.to_csv(ruta, index=False, encoding="utf-8")
        return str(ruta)

    def modo_puntual(self, ruta_csv: str | Path, estilo: str = "operador", intervalo: float = 5.0) -> str:
        if intervalo <= 0:
            raise ValueError("'intervalo' debe ser mayor que 0")

        df = self.cargar_csv(ruta_csv)
        nombre = Path(ruta_csv).stem

        logger.info("%s", "=" * 60)
        logger.info("MODO PUNTUAL — Analisis muestra a muestra")
        logger.info("%s", "=" * 60)

        tiempo_objetivo = float(df["tiempo_s"].iloc[0])
        filas_muestreadas: list[dict[str, Any]] = []
        i = 0

        while tiempo_objetivo <= float(df["tiempo_s"].iloc[-1]) + 1e-9:
            idx = (df["tiempo_s"] - tiempo_objetivo).abs().idxmin()
            fila = df.loc[idx]
            tendencia = 0.0

            if idx > 0:
                dt = df.loc[idx, "tiempo_s"] - df.loc[idx - 1, "tiempo_s"]
                if abs(dt) > 1e-12:
                    tendencia = (df.loc[idx, "nivel_m"] - df.loc[idx - 1, "nivel_m"]) / dt

            logger.info("Consulta puntual %d en t=%.2f s", i + 1, fila["tiempo_s"])
            prompt = self._construir_prompt_puntual(nombre, fila, tendencia, estilo)
            r = self.consultar_llm(prompt)
            estado = self.extraer_estado(r.respuesta) if r.exito else "ERROR_COMUNICACION"

            filas_muestreadas.append(
                {
                    **fila.to_dict(),
                    "tendencia_nivel_m_s": float(tendencia),
                    "estado": estado,
                    "estilo_salida": estilo,
                    "duracion_llm_s": r.duracion_s,
                    "exito_llm": r.exito,
                    "error_llm": r.error,
                    "respuesta_llm": r.respuesta,
                }
            )

            tiempo_objetivo += intervalo
            i += 1

        salida = self._guardar_resultados(filas_muestreadas, f"puntual_{nombre}_{estilo}.csv")
        logger.info("Resultados guardados: %s", salida)
        return salida

    def modo_periodo(self, ruta_csv: str | Path, desde: float, hasta: float, estilo: str = "operador") -> str:
        if hasta <= desde:
            raise ValueError("'hasta' debe ser mayor que 'desde'")

        df = self.cargar_csv(ruta_csv)
        nombre = Path(ruta_csv).stem
        bloque = df[(df["tiempo_s"] >= desde) & (df["tiempo_s"] <= hasta)].copy()
        if bloque.empty:
            raise ValueError("No hay datos en el rango solicitado")

        logger.info("%s", "=" * 60)
        logger.info("MODO PERIODO — Analisis agregado")
        logger.info("%s", "=" * 60)

        metricas = self.calcular_metricas(bloque)
        eventos = self._detectar_eventos(bloque)
        prompt = self._construir_prompt_periodo(nombre, desde, hasta, metricas, eventos, estilo)

        logger.info("Consultando LLM para analisis de periodo...")
        r = self.consultar_llm(prompt)
        estado = self.extraer_estado(r.respuesta) if r.exito else "ERROR_COMUNICACION"

        salida = self._guardar_resultados(
            {
                "archivo": nombre,
                "desde_s": desde,
                "hasta_s": hasta,
                "estado": estado,
                "estilo_salida": estilo,
                "duracion_llm_s": r.duracion_s,
                "exito_llm": r.exito,
                "error_llm": r.error,
                "respuesta_llm": r.respuesta,
                **metricas,
                "eventos": " | ".join(eventos) if eventos else "Sin eventos relevantes",
            },
            f"periodo_{nombre}_t{int(desde)}-{int(hasta)}_{estilo}.csv",
        )
        logger.info("Resultados guardados: %s", salida)
        return salida

    def modo_resumen(self, ruta_csv: str | Path, estilo: str = "operador") -> str:
        df = self.cargar_csv(ruta_csv)
        nombre = Path(ruta_csv).stem

        logger.info("%s", "=" * 60)
        logger.info("MODO RESUMEN — Analisis global")
        logger.info("%s", "=" * 60)

        metricas = self.calcular_metricas(df)
        mitad = max(len(df) // 2, 1)
        primera = df.iloc[:mitad].copy()
        segunda = df.iloc[mitad:].copy() if mitad < len(df) else df.iloc[:1].copy()

        metricas_1 = self.calcular_metricas(primera)
        metricas_2 = self.calcular_metricas(segunda)
        eventos = self._detectar_eventos(df)

        prompt = self._construir_prompt_resumen(nombre, metricas, metricas_1, metricas_2, eventos, estilo)
        logger.info("Consultando LLM para resumen global...")
        respuesta = self.consultar_llm(prompt)

        if not respuesta.exito:
            logger.error("Fallo al consultar el LLM: %s", respuesta.error)
            resultado = {
                "modo": "resumen",
                "archivo": nombre,
                "estado": "ERROR_COMUNICACION",
                "estilo_salida": estilo,
                "duracion_llm_s": respuesta.duracion_s,
                "error_llm": respuesta.error,
                "respuesta_llm": "",
                "eventos_detectados": len(eventos),
                **metricas,
                "mitad1_nivel_prom_m": metricas_1["nivel_prom_m"],
                "mitad1_error_abs_prom_m": metricas_1["error_abs_prom_m"],
                "mitad1_pct_normal": metricas_1["pct_normal"],
                "mitad1_pct_precaucion": metricas_1["pct_precaucion"],
                "mitad1_pct_alerta": metricas_1["pct_alerta"],
                "mitad2_nivel_prom_m": metricas_2["nivel_prom_m"],
                "mitad2_error_abs_prom_m": metricas_2["error_abs_prom_m"],
                "mitad2_pct_normal": metricas_2["pct_normal"],
                "mitad2_pct_precaucion": metricas_2["pct_precaucion"],
                "mitad2_pct_alerta": metricas_2["pct_alerta"],
                "eventos": " | ".join(eventos) if eventos else "Sin eventos relevantes",
            }
            ruta = self._guardar_resultados(resultado, f"resumen_{nombre}_{estilo}.csv")
            logger.info("Resultados guardados: %s", ruta)
            return ruta

        texto = respuesta.respuesta
        estado = self.extraer_estado(texto)
        print("-" * 60)
        print(texto)
        print("-" * 60)

        resultado = {
            "modo": "resumen",
            "archivo": nombre,
            "estado": estado,
            "estilo_salida": estilo,
            "duracion_llm_s": respuesta.duracion_s,
            "error_llm": "",
            "respuesta_llm": texto,
            "eventos_detectados": len(eventos),
            **metricas,
            "mitad1_nivel_prom_m": metricas_1["nivel_prom_m"],
            "mitad1_error_abs_prom_m": metricas_1["error_abs_prom_m"],
            "mitad1_pct_normal": metricas_1["pct_normal"],
            "mitad1_pct_precaucion": metricas_1["pct_precaucion"],
            "mitad1_pct_alerta": metricas_1["pct_alerta"],
            "mitad2_nivel_prom_m": metricas_2["nivel_prom_m"],
            "mitad2_error_abs_prom_m": metricas_2["error_abs_prom_m"],
            "mitad2_pct_normal": metricas_2["pct_normal"],
            "mitad2_pct_precaucion": metricas_2["pct_precaucion"],
            "mitad2_pct_alerta": metricas_2["pct_alerta"],
            "eventos": " | ".join(eventos) if eventos else "Sin eventos relevantes",
        }

        ruta = self._guardar_resultados(resultado, f"resumen_{nombre}_{estilo}.csv")
        logger.info("Resultados guardados: %s", ruta)
        return ruta

    def modo_comparar(self, ruta_csv_1: str | Path, ruta_csv_2: str | Path, estilo: str = "operador") -> str:
        df1 = self.cargar_csv(ruta_csv_1)
        df2 = self.cargar_csv(ruta_csv_2)
        nombre1 = Path(ruta_csv_1).stem
        nombre2 = Path(ruta_csv_2).stem

        m1 = self.calcular_metricas(df1)
        m2 = self.calcular_metricas(df2)
        prompt = self._construir_prompt_comparacion(nombre1, nombre2, m1, m2, estilo)

        logger.info("Consultando LLM para comparacion...")
        r = self.consultar_llm(prompt)
        estado = self.extraer_estado(r.respuesta) if r.exito else "ERROR_COMUNICACION"

        resultado = {
            "archivo_1": nombre1,
            "archivo_2": nombre2,
            "estado": estado,
            "estilo_salida": estilo,
            "duracion_llm_s": r.duracion_s,
            "exito_llm": r.exito,
            "error_llm": r.error,
            "respuesta_llm": r.respuesta,
            "esc1_nivel_prom_m": m1["nivel_prom_m"],
            "esc1_nivel_max_m": m1["nivel_max_m"],
            "esc1_error_abs_prom_m": m1["error_abs_prom_m"],
            "esc1_overshoot_pct": m1["overshoot_pct"],
            "esc1_control_prom_pct": m1["control_prom_pct"],
            "esc1_control_std_pct": m1["control_std_pct"],
            "esc1_pct_alerta": m1["pct_alerta"],
            "esc2_nivel_prom_m": m2["nivel_prom_m"],
            "esc2_nivel_max_m": m2["nivel_max_m"],
            "esc2_error_abs_prom_m": m2["error_abs_prom_m"],
            "esc2_overshoot_pct": m2["overshoot_pct"],
            "esc2_control_prom_pct": m2["control_prom_pct"],
            "esc2_control_std_pct": m2["control_std_pct"],
            "esc2_pct_alerta": m2["pct_alerta"],
            "delta_error_abs_prom_m": m2["error_abs_prom_m"] - m1["error_abs_prom_m"],
            "delta_pct_alerta": m2["pct_alerta"] - m1["pct_alerta"],
        }

        salida = self._guardar_resultados(resultado, f"comparacion_{nombre1}_vs_{nombre2}_{estilo}.csv")
        logger.info("Resultados guardados: %s", salida)
        return salida

    def modo_todos(self, carpeta: str | Path = ".", estilo: str = "operador") -> str:
        carpeta_path = Path(carpeta)
        if not carpeta_path.exists():
            raise FileNotFoundError(f"No existe la carpeta: {carpeta_path}")

        archivos = sorted(carpeta_path.glob("datos_escenario_*.csv"))
        if not archivos:
            raise FileNotFoundError(f"No se encontraron archivos 'datos_escenario_*.csv' en {carpeta_path.resolve()}")

        resumenes: list[dict[str, Any]] = []
        for archivo in archivos:
            logger.info("Procesando: %s", archivo.name)
            ruta_resumen = self.modo_resumen(archivo, estilo=estilo)
            try:
                df_res = pd.read_csv(ruta_resumen, encoding="utf-8")
                if not df_res.empty:
                    resumenes.append(df_res.iloc[0].to_dict())
            except Exception as exc:
                logger.warning("No fue posible incorporar '%s' al consolidado: %s", archivo.name, exc)

        salida = self._guardar_resultados(resumenes, f"resumen_consolidado_todos_{estilo}.csv")
        logger.info("Consolidado guardado: %s", salida)
        return salida


def pedir_archivo_existente(mensaje: str) -> str:
    while True:
        ruta = input(mensaje).strip().strip('"')
        if Path(ruta).exists():
            return ruta
        print("Archivo no encontrado. Intenta nuevamente.")


def pedir_float(mensaje: str, minimo: float | None = None) -> float:
    while True:
        valor = input(mensaje).strip().replace(",", ".")
        try:
            numero = float(valor)
            if minimo is not None and numero <= minimo:
                print(f"El valor debe ser mayor que {minimo}.")
                continue
            return numero
        except ValueError:
            print("Valor invalido. Ingresa un numero.")


def elegir_estilo() -> str:
    print("\nSelecciona estilo de respuesta:")
    print("1) Operador (corto y directo)")
    print("2) Tecnico (breve pero mas tecnico)")
    while True:
        opcion = input("Opcion: ").strip()
        if opcion in MODO_SALIDA:
            return MODO_SALIDA[opcion]
        print("Opcion invalida.")


def mostrar_menu() -> None:
    print("\nMenu de analisis")
    print("1) Analisis puntual")
    print("2) Analisis por periodo")
    print("3) Resumen global")
    print("4) Comparar dos CSV")
    print("5) Procesar todos los escenarios de una carpeta")
    print("0) Salir")


def ejecutar_menu_interactivo(orquestador: OrquestadorCDT) -> None:
    print("=" * 60)
    print("  GEMELO DIGITAL COGNITIVO")
    print("  Orquestador interactivo v4.0")
    print("  AnythingLLM + Qwen")
    print("=" * 60)

    if not orquestador.verificar_conexion():
        raise SystemExit(1)

    csv_principal = pedir_archivo_existente("\nIngresa la ruta del CSV principal: ")
    estilo = elegir_estilo()

    while True:
        mostrar_menu()
        opcion = input("Selecciona una opcion: ").strip()

        if opcion not in MODO_ANALISIS:
            print("Opcion invalida.")
            continue

        if opcion == "0":
            print("Saliendo del programa.")
            break

        try:
            if opcion == "1":
                intervalo = pedir_float("Intervalo de muestreo [s] (ej. 5): ", minimo=0)
                ruta = orquestador.modo_puntual(csv_principal, estilo=estilo, intervalo=intervalo)
            elif opcion == "2":
                desde = pedir_float("Tiempo inicial [s]: ")
                hasta = pedir_float("Tiempo final [s]: ")
                ruta = orquestador.modo_periodo(csv_principal, desde=desde, hasta=hasta, estilo=estilo)
            elif opcion == "3":
                ruta = orquestador.modo_resumen(csv_principal, estilo=estilo)
            elif opcion == "4":
                csv2 = pedir_archivo_existente("Ingresa la ruta del segundo CSV: ")
                ruta = orquestador.modo_comparar(csv_principal, csv2, estilo=estilo)
            elif opcion == "5":
                carpeta = input("Ingresa carpeta de escenarios [Enter para actual]: ").strip() or "."
                ruta = orquestador.modo_todos(carpeta, estilo=estilo)
            else:
                continue

            print("\n" + "=" * 60)
            print("Analisis finalizado")
            print(f"Resultado(s): {ruta}")
            print("=" * 60)

        except Exception as exc:
            print(f"\nError: {exc}")

        if input("\n¿Quieres cambiar el estilo de salida? [s/N]: ").strip().lower() == "s":
            estilo = elegir_estilo()

        if input("¿Quieres cambiar el CSV principal? [s/N]: ").strip().lower() == "s":
            csv_principal = pedir_archivo_existente("Nueva ruta de CSV principal: ")


def main() -> None:
    orquestador = OrquestadorCDT(api_key=API_KEY, url=ANYTHINGLLM_URL, workspace_slug=WORKSPACE_SLUG)
    ejecutar_menu_interactivo(orquestador)


if __name__ == "__main__":
    main()
