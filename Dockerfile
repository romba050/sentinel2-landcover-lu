# Serves the static web demo (web/) -- the deployable artifact of this repo.
# Built on the server by the infra repo's deploy.sh:
#   docker compose up -d --build landcover
FROM nginx:alpine
COPY web/ /usr/share/nginx/html/
