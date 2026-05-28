# Node.js Web Template

Minimal Node.js starter app for running behind Apache on HTTPS.

## Structure

- `src/server.js` — Express server entrypoint
- `public/index.html` — starter landing page
- `.env.example` — local environment defaults
- `package.json` — scripts and dependencies

## Quick start

1. From `web`, install dependencies:
	- `npm install`
2. Copy the environment file if needed:
	- `cp .env.example .env`
3. Run in development:
	- `npm run dev`
4. Run in production:
	- `npm start`
5. Or use the bootstrap script:
	- `bash install_and_run.sh`

The bootstrap script detects macOS or Linux, installs `node` and `npm` if they are missing, installs project dependencies, creates `.env` from `.env.example` when needed, and starts the app.

By default the app listens on `127.0.0.1:3000`, which is a good fit for Apache reverse proxying from port `443`.

## Available routes

- `/` — starter HTML page
- `/health` — simple health check
- `/api/hello` — JSON example endpoint

## Apache reverse proxy example

Enable the required modules:

- `proxy`
- `proxy_http`
- `headers`
- `rewrite`

Example HTTPS virtual host snippet:

```apache
<VirtualHost *:443>
	 ServerName rigoroushuman.com

	 SSLEngine on
	 SSLCertificateFile /path/to/fullchain.pem
	 SSLCertificateKeyFile /path/to/privkey.pem

	 ProxyPreserveHost On
	 ProxyRequests Off

	 RequestHeader set X-Forwarded-Proto "https"
	 RequestHeader set X-Forwarded-Port "443"

	 ProxyPass / http://127.0.0.1:3000/
	 ProxyPassReverse / http://127.0.0.1:3000/
</VirtualHost>
```

## Notes

- Keep Apache exposed publicly on `443`.
- Keep Node.js bound to localhost unless you intentionally want public access.
- For long-running production use, run the app with a process manager such as `systemd`, `pm2`, or Docker.
