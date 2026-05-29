const http = require('http');
const path = require('path');

require('dotenv').config();

const express = require('express');

const app = express();
const host = process.env.HOST || '127.0.0.1';
const port = Number(process.env.PORT) || 3000;
const trustProxy = String(process.env.TRUST_PROXY || 'true').toLowerCase() === 'true';
const appName = process.env.APP_NAME || 'Rigorous Human';
const keepAliveTimeout = Number(process.env.KEEP_ALIVE_TIMEOUT) || 65000;
const headersTimeout = Number(process.env.HEADERS_TIMEOUT) || 66000;

let isReady = false;

app.disable('x-powered-by');
app.set('trust proxy', trustProxy);

app.use(express.json());
app.use(express.static(path.join(__dirname, '..', 'public')));

app.get('/health', (_request, response) => {
  const payload = {
    status: isReady ? 'ok' : 'starting',
    uptime: process.uptime(),
    timestamp: new Date().toISOString()
  };

  if (!isReady) {
    response.status(503).json(payload);
    return;
  }

  response.json(payload);
});

app.get('/api/hello', (request, response) => {
  response.json({
    message: `Hello from ${appName}`,
    secure: request.secure,
    host: request.get('host')
  });
});

const server = http.createServer(app);
server.keepAliveTimeout = keepAliveTimeout;
server.headersTimeout = headersTimeout;

server.listen(port, host, () => {
  isReady = true;
  console.log(`${appName} listening on http://${host}:${port}`);
});

server.on('error', (error) => {
  console.error('Server error:', error);
});

const shutdown = (signal) => {
  console.log(`Received ${signal}, shutting down...`);
  isReady = false;
  server.close(() => {
    process.exit(0);
  });

  setTimeout(() => {
    process.exit(1);
  }, 10000).unref();
};

process.on('SIGTERM', () => shutdown('SIGTERM'));
process.on('SIGINT', () => shutdown('SIGINT'));
process.on('uncaughtException', (error) => {
  console.error('Uncaught exception:', error);
  shutdown('uncaughtException');
});
process.on('unhandledRejection', (reason) => {
  console.error('Unhandled rejection:', reason);
  shutdown('unhandledRejection');
});