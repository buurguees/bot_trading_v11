# Configuración del Entorno Virtual - Bot Trading v11

## 1. Crear el Entorno Virtual

```powershell
# Navegar al directorio del proyecto
cd C:\Users\Alex B\Desktop\bot_trading_v9\bot_trading_v11

# Crear entorno virtual en la raíz del proyecto
python -m venv venv
```

## 2. Activar el Entorno Virtual

```powershell
# Windows (PowerShell)
.\venv\Scripts\activate

# Windows (CMD)
venv\Scripts\activate.bat

# Verificar que está activo (debería mostrar (venv) al inicio)
python --version
```

## 3. Instalar Dependencias

```powershell
# Asegurar que pip esté actualizado
python -m pip install --upgrade pip

# Instalar dependencias desde requirements.txt
pip install -r requirements.txt
```

## 3.1. Configurar Variables de Entorno

```powershell
# Copiar archivo de ejemplo de configuración
copy config\env.example config\.env

# Editar el archivo .env con tus credenciales reales
notepad config\.env
```

**Importante**: 
- El archivo `config/.env` contiene credenciales sensibles y está en `.gitignore` para no subirse al repositorio
- Todos los scripts del sistema buscan automáticamente `config/.env` como archivo principal de configuración
- Si `config/.env` no existe, los scripts usarán `config/env.example` como fallback

## 3.2. Configurar Base de Datos PostgreSQL

```powershell
# Instalar PostgreSQL (si no está instalado)
# Descargar desde: https://www.postgresql.org/download/windows/

# Inicializar base de datos
python scripts\initialization\init_db.py

# Verificar configuración
python scripts\initialization\verify_db.py
```

**Nota**: Asegúrate de que PostgreSQL esté ejecutándose antes de ejecutar los scripts.

## 4. Instalación Especial para TA-Lib (Windows)

TA-Lib requiere instalación previa del binario:

### Opción A: Instalación directa (Recomendada)
```powershell
# TA-Lib ya está disponible para Python 3.13+ en PyPI
pip install TA-Lib
```

### Opción B: Instalación manual
1. Descargar `TA_Lib-0.4.28-cp310-cp310-win_amd64.whl` desde: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
2. Instalar: `pip install TA_Lib-0.4.28-cp310-cp310-win_amd64.whl`

## 5. Verificar Instalación

```powershell
# Verificar que todas las dependencias se instalaron correctamente
pip list

# Probar importaciones críticas
python -c "import ccxt, sqlalchemy, stable_baselines3, talib, telegram"
```

## 6. Actualizar Dependencias

```powershell
# Si modificas requirements.txt o necesitas actualizar
pip install -r requirements.txt --upgrade

# Actualizar una dependencia específica
pip install --upgrade ccxt
```

## 7. Desactivar el Entorno Virtual

```powershell
# Cuando termines de trabajar
deactivate
```

## 8. Comandos Útiles

```powershell
# Ver dependencias instaladas
pip freeze

# Generar requirements.txt desde el entorno actual
pip freeze > requirements_current.txt

# Instalar dependencias de desarrollo
pip install -r requirements.txt[dev]

# Limpiar caché de pip
pip cache purge
```

## 9. Solución de Problemas Comunes

### Error con TA-Lib:
```powershell
# Si falla la instalación de TA-Lib, instalar Visual C++ Build Tools
# Descargar desde: https://visualstudio.microsoft.com/visual-cpp-build-tools/
```

### Error con psycopg2:
```powershell
# Instalar dependencias del sistema
pip install psycopg2-binary
```

### Error de permisos:
```powershell
# Ejecutar PowerShell como Administrador
# O usar --user para instalar en el directorio del usuario
pip install --user -r requirements.txt
```

## 10. Estructura del Entorno

```
bot_trading_v11/
├── venv/                    # Entorno virtual
│   ├── Scripts/            # Scripts de activación
│   ├── Lib/                # Librerías Python
│   └── pyvenv.cfg          # Configuración del entorno
├── requirements.txt        # Dependencias del proyecto
└── setup_environment.md    # Este archivo
```
