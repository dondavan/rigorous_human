const path = require('path');

require('dotenv').config();

const express = require('express');

const app = express();
const host = process.env.HOST || '127.0.0.1';
const port = Number(process.env.PORT) || 3000;
const trustProxy = String(process.env.TRUST_PROXY || 'true').toLowerCase() === 'true';
const appName = process.env.APP_NAME || 'Rigorous Human';

app.disable('x-powered-by');
app.set('trust proxy', trustProxy);

app.use(express.json());
app.use(express.static(path.join(__dirname, '..', 'public')));

app.get('/health', (_request, response) => {
  response.json({ status: 'ok' });
});

app.get('/api/hello', (request, response) => {
  response.json({
    message: `Hello from ${appName}`,
    secure: request.secure,
    host: request.get('host')
  });
});

app.listen(port, host, () => {
  console.log(`${appName} listening on http://${host}:${port}`);
});