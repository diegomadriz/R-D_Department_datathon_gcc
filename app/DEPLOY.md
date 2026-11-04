# Despliegue de la app en Vercel

Guía desde cero. No necesitas conocer el proyecto, ni Python, ni Databricks. Toma unos 15 minutos.

La app ya está construida. Tu trabajo es publicarla en Vercel, protegida con contraseña, y comprobar que funciona. No hay que cambiar código.

> **Antes de empezar:** confirma con el equipo que el diseño ya se validó y qué rama usar. Esta guía asume `bi/f5-app`. Si ya se fusionó, usa `main`.

## Qué es la app

Una página web que muestra resultados del modelo de resistencia del cemento. Todo lo que necesita ya está en `app/public/`:

- `index.html`: la interfaz.
- `snapshot.js`: los datos, solo agregados, sin filas individuales.

`app/src/proxy.ts` pide la contraseña antes de entregar cualquier archivo. La contraseña vive solo en una variable de entorno de Vercel: `APP_PASSWORD`.

## 1. Lo que necesitas

| Requisito | Cómo comprobarlo |
|---|---|
| Acceso al repo privado `diegomadriz/R-D_Department_datathon_gcc` | Pídelo al equipo si el paso 2 falla |
| Git | `git --version` |
| Node.js 20 o superior (probado con 24) | `node -v`; si falta, instálalo desde nodejs.org |
| Cuenta de Vercel | Créala en vercel.com |

Acuerda con el equipo en qué cuenta de Vercel va a quedar la app. El plan gratuito (Hobby) restringe el uso comercial; nadie ha verificado todavía si este caso le aplica, así que coméntalo si la cuenta es personal.

## 2. Descarga el código

```bash
git clone https://github.com/diegomadriz/R-D_Department_datathon_gcc.git
```

```bash
cd R-D_Department_datathon_gcc
```

```bash
git switch bi/f5-app
```

```bash
cd app
```

**Todos los comandos siguientes se ejecutan dentro de `app/`.** Si los corres en la raíz del repo, Vercel no detectará la app.

## 3. Comprueba que compila

```bash
npm ci
```

```bash
npx next build
```

Debe terminar sin errores y mostrar una línea `Proxy (Middleware)`. Si no aparece, no sigas y avisa al equipo: la contraseña no estaría activa.

## 4. Conecta con Vercel

```bash
npx vercel@65.11.0 login
```

```bash
npx vercel@65.11.0 link
```

Responde a las preguntas de `link`:

- **Set up and link?** Sí.
- **Scope:** la cuenta acordada.
- **Link to existing project?** No.
- **Project name:** `gcc-resistencia`, o el que acuerden.
- **Directory:** `./`
- **Modify settings?** No. Vercel debe detectar Next.js.
- Si pregunta por conectar el repositorio de GitHub, responde que **no**: esta guía despliega desde tu computadora.

Esto crea la carpeta `.vercel/`, que ya está ignorada por git.

## 5. Define la contraseña

**Hazlo antes de desplegar.** Sin ella, la app responde `503` y no muestra nada.

Genera una contraseña larga:

```bash
openssl rand -base64 24
```

Guárdala en tu gestor de contraseñas y regístrala en Vercel. El comando te la pedirá; pégala ahí:

```bash
npx vercel@65.11.0 env add APP_PASSWORD --environment production --sensitive
```

No la pongas en ningún archivo, commit, issue ni chat grupal.

## 6. Publica

```bash
npx vercel@65.11.0 --prod
```

Al terminar muestra la URL de producción, normalmente `https://<proyecto>.vercel.app`. **Esa es la que se comparte**, no la URL larga de ese despliegue en particular.

## 7. Comprueba que quedó bien

Sustituye `<URL>` por tu URL de producción.

Sin contraseña debe responder `401`, tanto la página como los datos:

```bash
curl -s -o /dev/null -w '%{http_code}\n' <URL>/
```

```bash
curl -s -o /dev/null -w '%{http_code}\n' <URL>/snapshot.js
```

Con contraseña debe responder `200`. El comando te la pedirá:

```bash
curl -s -o /dev/null -w '%{http_code}\n' -u visor <URL>/
```

Después ábrela en una ventana privada del navegador. El usuario puede ser cualquiera, por ejemplo `visor`. Revisa que:

- [ ] Pide usuario y contraseña antes de mostrar algo.
- [ ] Abren todas las páginas del menú superior.
- [ ] En **Simulador**, al mover un control cambian las cifras y la curva.
- [ ] En **Resistencias** se ven gráficas con datos y funciona el filtro por tipo.
- [ ] Se ve bien en un teléfono o en una ventana angosta.
- [ ] La consola del navegador (F12) no muestra errores en rojo.

## 8. Entrega

Envía la URL y la contraseña por un canal privado a quien coordina la entrega. Confirma que **probaste la URL en una ventana privada** y avisa de cualquier casilla que no haya pasado.

## Si algo falla

| Síntoma | Causa probable | Qué hacer |
|---|---|---|
| `503` con `APP_PASSWORD is not configured` | Falta la variable, o se agregó después de desplegar | Repite el paso 5 y luego el 6 |
| La contraseña correcta da `401` | Espacios o saltos de línea al pegarla | Bórrala con `npx vercel@65.11.0 env rm APP_PASSWORD`, repite el paso 5 y luego el 6 |
| **Sin contraseña da `200`** | La protección no está activa | **Retira la app de inmediato** (fila siguiente) y avisa al equipo |
| Hay que retirar o volver atrás | Algo se publicó mal | `npx vercel@65.11.0 rollback`, o bórrala con `npx vercel@65.11.0 project rm <proyecto>` |
| Vercel te pide iniciar sesión en vez de la contraseña | Protección de despliegues de Vercel, o abriste la URL larga | Usa la URL de producción. Si sigue, revisa **Settings → Deployment Protection** en el proyecto |
| `404` en todas las páginas | Se desplegó desde la raíz del repo | Ejecuta `cd app` y repite desde el paso 4 |
| Falla `npm ci` o el build | Versión de Node o rama equivocada | Revisa `node -v` y `git branch --show-current` |
| La página abre, pero sin cifras | No cargó `snapshot.js` | Revisa la pestaña Red del navegador y avisa al equipo |

## Lo que no debes hacer

- No cambies `app/public/snapshot.js` a mano ni agregues archivos de datos: la app publica todo lo que haya en `public/`.
- No subas al repo `.env`, `.vercel/` ni la contraseña.
- No quites `src/proxy.ts`.
- Regenerar los datos no es parte de este despliegue, y no necesitas Databricks.

## Cambiar la contraseña después

Borra la variable, vuelve a crearla y despliega de nuevo. El cambio solo aplica al nuevo despliegue:

```bash
npx vercel@65.11.0 env rm APP_PASSWORD
```

Luego repite los pasos 5, 6 y 7.
