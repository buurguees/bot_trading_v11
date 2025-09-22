-- =====================================================
-- CONSULTAS PARA INSPECCIONAR LA BASE DE DATOS
-- =====================================================
-- Script con consultas útiles para explorar la estructura de la BD
-- =====================================================

-- 1. LISTAR TODAS LAS TABLAS EN EL ESQUEMA TRADING
-- =====================================================
SELECT 
    table_name as "Tabla",
    pg_size_pretty(pg_total_relation_size('trading.' || table_name)) as "Tamaño"
FROM information_schema.tables 
WHERE table_schema = 'trading'
ORDER BY pg_total_relation_size('trading.' || table_name) DESC;

-- 2. VER TODAS LAS COLUMNAS DE UNA TABLA ESPECÍFICA
-- =====================================================
-- (Reemplaza 'NOMBRE_TABLA' con el nombre de la tabla que quieres inspeccionar)
SELECT 
    column_name as "Columna",
    data_type as "Tipo",
    is_nullable as "Permite NULL",
    column_default as "Valor por Defecto",
    character_maximum_length as "Longitud Máxima"
FROM information_schema.columns 
WHERE table_schema = 'trading' 
  AND table_name = 'NOMBRE_TABLA'  -- 👈 Cambia aquí el nombre de la tabla
ORDER BY ordinal_position;

-- 3. VER TODOS LOS ÍNDICES DE UNA TABLA
-- =====================================================
SELECT 
    indexname as "Nombre del Índice",
    indexdef as "Definición del Índice"
FROM pg_indexes 
WHERE schemaname = 'trading' 
  AND tablename = 'NOMBRE_TABLA'  -- 👈 Cambia aquí el nombre de la tabla
ORDER BY indexname;

-- 4. ESTADÍSTICAS DE TODAS LAS TABLAS
-- =====================================================
SELECT 
    relname as "Tabla",
    n_tup_ins as "Inserts",
    n_tup_upd as "Updates", 
    n_tup_del as "Deletes",
    n_live_tup as "Tuplas Vivas",
    n_dead_tup as "Tuplas Muertas",
    last_vacuum as "Último VACUUM",
    last_analyze as "Último ANALYZE"
FROM pg_stat_user_tables 
WHERE schemaname = 'trading'
ORDER BY n_live_tup DESC;

-- 5. VER CONSTRAINTS (RESTRICCIONES) DE UNA TABLA
-- =====================================================
SELECT 
    constraint_name as "Nombre de Restricción",
    constraint_type as "Tipo",
    check_clause as "Condición"
FROM information_schema.table_constraints tc
LEFT JOIN information_schema.check_constraints cc 
    ON tc.constraint_name = cc.constraint_name
WHERE tc.table_schema = 'trading' 
  AND tc.table_name = 'NOMBRE_TABLA'  -- 👈 Cambia aquí el nombre de la tabla
ORDER BY constraint_name;

-- 6. VER TRIGGERS DE UNA TABLA
-- =====================================================
SELECT 
    trigger_name as "Nombre del Trigger",
    event_manipulation as "Evento",
    action_timing as "Momento",
    action_statement as "Acción"
FROM information_schema.triggers 
WHERE event_object_schema = 'trading' 
  AND event_object_table = 'NOMBRE_TABLA'  -- 👈 Cambia aquí el nombre de la tabla
ORDER BY trigger_name;

-- 7. VER PARTICIONES (si las hay)
-- =====================================================
SELECT 
    schemaname as "Esquema",
    tablename as "Tabla Padre",
    partitionname as "Partición",
    partitionbounddef as "Definición de Partición"
FROM pg_partitions 
WHERE schemaname = 'trading'
ORDER BY tablename, partitionname;

-- 8. VER SECUENCIAS (SERIAL/BIGSERIAL)
-- =====================================================
SELECT 
    sequence_name as "Nombre de Secuencia",
    data_type as "Tipo",
    start_value as "Valor Inicial",
    minimum_value as "Valor Mínimo",
    maximum_value as "Valor Máximo",
    increment as "Incremento"
FROM information_schema.sequences 
WHERE sequence_schema = 'trading'
ORDER BY sequence_name;

-- 9. VER FUNCIONES Y PROCEDIMIENTOS
-- =====================================================
SELECT 
    routine_name as "Nombre",
    routine_type as "Tipo",
    data_type as "Tipo de Retorno",
    routine_definition as "Definición"
FROM information_schema.routines 
WHERE routine_schema = 'trading'
ORDER BY routine_name;

-- 10. CONSULTA RÁPIDA: VER ESTRUCTURA COMPLETA DE UNA TABLA
-- =====================================================
-- Esta consulta combina columnas, tipos, índices y estadísticas
WITH table_info AS (
    SELECT 
        c.table_name,
        c.column_name,
        c.data_type,
        c.is_nullable,
        c.column_default,
        c.ordinal_position
    FROM information_schema.columns c
    WHERE c.table_schema = 'trading' 
      AND c.table_name = 'NOMBRE_TABLA'  -- 👈 Cambia aquí el nombre de la tabla
),
index_info AS (
    SELECT 
        i.tablename,
        i.indexname,
        i.indexdef
    FROM pg_indexes i
    WHERE i.schemaname = 'trading' 
      AND i.tablename = 'NOMBRE_TABLA'  -- 👈 Cambia aquí el nombre de la tabla
),
stats_info AS (
    SELECT 
        s.relname,
        s.n_live_tup,
        s.n_dead_tup,
        s.last_vacuum,
        s.last_analyze
    FROM pg_stat_user_tables s
    WHERE s.schemaname = 'trading' 
      AND s.relname = 'NOMBRE_TABLA'  -- 👈 Cambia aquí el nombre de la tabla
)
SELECT 
    'COLUMNAS' as tipo,
    ti.column_name as nombre,
    ti.data_type as detalle,
    ti.is_nullable as extra
FROM table_info ti
UNION ALL
SELECT 
    'INDICES' as tipo,
    ii.indexname as nombre,
    'BRIN' as detalle,
    CASE WHEN ii.indexdef LIKE '%USING BRIN%' THEN 'BRIN' ELSE 'B-tree' END as extra
FROM index_info ii
UNION ALL
SELECT 
    'ESTADISTICAS' as tipo,
    'Tuplas vivas' as nombre,
    si.n_live_tup::text as detalle,
    'Último VACUUM: ' || COALESCE(si.last_vacuum::text, 'Nunca') as extra
FROM stats_info si
ORDER BY tipo, nombre;

-- 11. VER USUARIOS Y PERMISOS
-- =====================================================
SELECT 
    usename as "Usuario",
    usesuper as "Es Superusuario",
    usecreatedb as "Puede Crear BD",
    usebypassrls as "Bypass RLS"
FROM pg_user 
WHERE usename LIKE '%trading%'
ORDER BY usename;

-- 12. VER CONEXIONES ACTIVAS
-- =====================================================
SELECT 
    pid as "PID",
    usename as "Usuario",
    application_name as "Aplicación",
    client_addr as "IP Cliente",
    state as "Estado",
    query_start as "Inicio de Query",
    LEFT(query, 50) as "Query (primeros 50 chars)"
FROM pg_stat_activity 
WHERE datname = 'trading_db'
  AND state = 'active'
ORDER BY query_start;

-- 13. VER TAMAÑO DE LA BASE DE DATOS COMPLETA
-- =====================================================
SELECT 
    pg_size_pretty(pg_database_size('trading_db')) as "Tamaño Total de la BD";

-- 14. VER ESPACIO USADO POR ESQUEMA
-- =====================================================
SELECT 
    schemaname as "Esquema",
    pg_size_pretty(SUM(pg_total_relation_size(schemaname||'.'||tablename))) as "Tamaño Total"
FROM pg_tables 
WHERE schemaname = 'trading'
GROUP BY schemaname
ORDER BY SUM(pg_total_relation_size(schemaname||'.'||tablename)) DESC;
