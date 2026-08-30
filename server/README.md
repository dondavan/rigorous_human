# Web Design
Apache2 as frontend to redirect request, Node.js as serving


Apache:
Redirect ip + sub-domain name to HTTPs, see server/apache2.conf
Aka only :443 is the entry point, forward requet to internal port 3000

Node.js:

Deployment checklist:

1. Put the `VirtualHost *:443` block from `server/apache2.conf` into the active Apache SSL site.
2. Replace `/path/to/fullchain.pem` and `/path/to/privkey.pem` with the real certificate paths.
3. Enable the Apache modules `proxy`, `proxy_http`, `headers`, `rewrite`, and `ssl`.
4. Reload Apache after editing the vhost.
5. Make sure no other enabled SSL site on `:443` is taking precedence.




# Web set-up
1. HTTPS:SSL:certi-bot
2. Redirect HTTP to HTTPS: /etc/apache2/apache2.conf
3. Proxy setup at          /etc/apache2/sites-available/default-ssl.conf



# Certificate
HTTPs Certification are managed through certbot

Renewal:
> sudo certbot renew