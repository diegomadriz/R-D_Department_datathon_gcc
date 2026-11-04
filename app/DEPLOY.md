# Despliegue de la app en Vercel

Guía desde cero. No necesitas conocer el proyecto, ni Python, ni Databricks. Toma unos 15 minutos.

La app ya está construida. Tu trabajo es publicarla en Vercel y comprobar que funciona. No hay que cambiar código.

> **Antes de empezar:** confirma con el equipo que el diseño ya se validó y qué rama usar. Esta guía usa `main`.

## Qué es la app

Una página web que muestra resultados del modelo de resistencia del cemento. Todo lo que necesita ya está en `app/public/`:

- `index.html`: la interfaz.
- `snapshot.js`: los datos, solo agregados, sin filas individuales.
- `demo_gcc.mp4`: el video de la página Demo.

La app es de acceso abierto: cualquiera con el enlace puede verla. Se pide a los buscadores no indexarla mediante `X-Robots-Tag`, lo que no impide el acceso.

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

Debe terminar sin errores.

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

## 5. Publica

```bash
npx vercel@65.11.0 --prod
```

Al terminar muestra la URL de producción, normalmente `https://<proyecto>.vercel.app`. **Esa es la que se comparte**, no la URL larga de ese despliegue en particular.

## 6. Comprueba que quedó bien

Sustituye `<URL>` por tu URL de producción. Los tres comandos deben responder `200`:

```bash
curl -s -o /dev/null -w '%{http_code}\n' <URL>/
```

```bash
curl -s -o /dev/null -w '%{http_code}\n' <URL>/snapshot.js
```

```bash
curl -s -o /dev/null -w '%{http_code}\n' <URL>/demo_gcc.mp4
```

Después ábrela en una ventana privada del navegador. Revisa que:

- [ ] Abren todas las páginas del menú superior.
- [ ] En **Simulador**, al mover un control cambian las cifras y la curva.
- [ ] En **Resistencias** se ven gráficas con datos y funciona el filtro por tipo.
- [ ] En **Demo** el video reproduce y permite saltar.
- [ ] Se ve bien en un teléfono o en una ventana angosta.
- [ ] La consola del navegador (F12) no muestra errores en rojo.

## 7. Entrega

Envía la URL a quien coordina la entrega. Confirma que **probaste la URL en una ventana privada** y avisa de cualquier casilla que no haya pasado.

## Si algo falla

| Síntoma | Causa probable | Qué hacer |
|---|---|---|
| Hay que retirar o volver atrás | Algo se publicó mal | `npx vercel@65.11.0 rollback`, o bórrala con `npx vercel@65.11.0 project rm <proyecto>` |
| Vercel te pide iniciar sesión | Protección de despliegues de Vercel, o abriste la URL larga | Usa la URL de producción. Si sigue, revisa **Settings → Deployment Protection** en el proyecto |
| `404` en todas las páginas | Se desplegó desde la raíz del repo | Ejecuta `cd app` y repite desde el paso 4 |
| Falla `npm ci` o el build | Versión de Node o rama equivocada | Revisa `node -v` y `git branch --show-current` |
| La página abre, pero sin cifras | No cargó `snapshot.js` | Revisa la pestaña Red del navegador y avisa al equipo |

## Lo que no debes hacer

- No cambies `app/public/snapshot.js` a mano ni agregues archivos de datos: la app publica sin restricción todo lo que haya en `public/`.
- No subas al repo `.env` ni `.vercel/`.
- Regenerar los datos no es parte de este despliegue, y no necesitas Databricks.
