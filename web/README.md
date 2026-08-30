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

If the browser still shows an Apache page instead of this app, the usual cause is that the active SSL vhost is not proxying to `127.0.0.1:3000` or another `:443` site is taking precedence. Make sure the `VirtualHost *:443` block from `server/apache2.conf` is the enabled HTTPS site and reload Apache.

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

## Production deployment

### Option 1: Daemon script (simple, no additional dependencies)

Run the service in the background with automatic health checks and restart-on-crash:

```bash
bash run-daemon.sh &
```

The script:
- Monitors the `/health` endpoint every 10 seconds
- Automatically restarts if the service crashes
- Restarts after 3 consecutive health check failures
- Logs all activity to `logs/service.log`
- Can be stopped with: `kill $(cat .daemon.pid)`

### Option 2: systemd service (production-grade, GCP recommended)

1. Edit `rigorous-human.service` and update the `WorkingDirectory` path to match your deployment.
2. Copy the service file to systemd:
   ```bash
   sudo cp rigorous-human.service /etc/systemd/system/
   ```
3. Enable and start:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable rigorous-human
   sudo systemctl start rigorous-human
   ```
4. Check status:
   ```bash
   sudo systemctl status rigorous-human
   ```
5. View logs:
   ```bash
   sudo journalctl -u rigorous-human -f
   ```

## Notes

- Keep Apache exposed publicly on `443`.
- Keep Node.js bound to localhost unless you intentionally want public access.
- The `/health` endpoint returns `503` until the app is ready, then `200` with uptime info—use this for load balancer checks.
- Daemon script and systemd service both auto-restart on crash; use one or the other, not both.
