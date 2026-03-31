function exportar_csv(nombre_escenario)
%% EXPORTAR_CSV  Exporta datos de Simulink al formato CSV del orquestador
%
%  Uso: Después de correr sim('estanque_pid'), ejecutar:
%    exportar_csv('escenario_1')
%
%  Genera: datos_escenario_1.csv
%
%  Las variables se leen automáticamente del workspace.

    % Parámetros físicos
    Cd = 0.6;
    a_salida = 2e-4;
    g = 9.81;

    % Leer variables del workspace
    t  = evalin('base', 'tout');
    
    % Nivel (h)
    h_raw = evalin('base', 'simout_h');
    if isa(h_raw, 'timeseries'), h = h_raw.Data;
    elseif isstruct(h_raw), h = h_raw.signals.values;
    else, h = h_raw; end
    
    % Setpoint
    SP_raw = evalin('base', 'simout_SP');
    if isa(SP_raw, 'timeseries'), SP = SP_raw.Data;
    elseif isstruct(SP_raw), SP = SP_raw.signals.values;
    else, SP = SP_raw; end
    
    % Señal de control (%)
    u_raw = evalin('base', 'simout_u');
    if isa(u_raw, 'timeseries'), u = u_raw.Data;
    elseif isstruct(u_raw), u = u_raw.signals.values;
    else, u = u_raw; end
    
    % Qin
    try
        Qin_raw = evalin('base', 'simout_Qin');
        if isa(Qin_raw, 'timeseries'), Qin = Qin_raw.Data;
        elseif isstruct(Qin_raw), Qin = Qin_raw.signals.values;
        else, Qin = Qin_raw; end
    catch
        Qin_max = 3e-4;
        Qin = (u / 100) * Qin_max;
        fprintf('  NOTA: Qin estimado desde señal de control\n');
    end
    
    % Error (calculado, no leído, para garantizar consistencia)
    e = SP - h;
    
    % Qout por Torricelli
    Qout = Cd * a_salida * sqrt(2 * g * max(h, 0));
    
    % Igualar longitudes (por si alguna señal tiene distinto largo)
    n = min([length(t), length(h), length(SP), length(u), length(Qin)]);
    t = t(1:n);
    h = h(1:n);
    SP = SP(1:n);
    e = e(1:n);
    Qin = Qin(1:n);
    Qout = Qout(1:n);
    u = u(1:n);
    
    % Crear tabla con los nombres exactos que espera el orquestador
    T = table(t, h, SP, e, Qin, Qout, u, ...
        'VariableNames', {'tiempo_s', 'nivel_m', 'setpoint_m', ...
        'error_m', 'Qin_m3s', 'Qout_m3s', 'control_pct'});
    
    % Exportar
    nombre_archivo = ['datos_' nombre_escenario '.csv'];
    writetable(T, nombre_archivo);
    
    % Resumen
    fprintf('\n========================================\n');
    fprintf('  CSV exportado: %s\n', nombre_archivo);
    fprintf('========================================\n');
    fprintf('  Filas:       %d\n', height(T));
    fprintf('  Duracion:    %.1f s\n', t(end));
    fprintf('  Nivel min:   %.4f m\n', min(h));
    fprintf('  Nivel max:   %.4f m\n', max(h));
    fprintf('  Nivel final: %.4f m\n', h(end));
    fprintf('  SP final:    %.4f m\n', SP(end));
    fprintf('  Error final: %.4f m\n', e(end));
    fprintf('========================================\n\n');
    
    % Gráfico rápido de verificación
    figure('Name', ['Verificación - ' nombre_escenario]);
    subplot(2,1,1);
    plot(t, h, 'b-', 'LineWidth', 1.5); hold on;
    plot(t, SP, 'r--', 'LineWidth', 1);
    ylabel('Nivel [m]');
    legend('h(t)', 'SP');
    title(['Escenario: ' nombre_escenario]);
    grid on;
    
    subplot(2,1,2);
    plot(t, u, 'g-', 'LineWidth', 1);
    ylabel('Control [%]');
    xlabel('Tiempo [s]');
    title('Señal de control PID');
    grid on;
    
    fprintf('Gráfico de verificación generado.\n');
    fprintf('Si h(t) converge al SP, los datos son correctos.\n\n');
end
