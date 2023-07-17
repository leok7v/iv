@echo off
if not exist "%1" (
  echo downloading "%1" from "%2"
  rem cURL --ssl-no-revoke is there because github CDN certificate is revoked
  rem cURL --location is actually "follow redirect"
  curl.exe --ssl-no-revoke --silent --location "%2" --create-dirs --output "%1"
)