ARG NODE_IMAGE=registry.cn-hangzhou.aliyuncs.com/migo-dl/node:22-alpine
ARG NGINX_IMAGE=registry.cn-hangzhou.aliyuncs.com/migo-dl/nginx:1.27-alpine
ARG PNPM_REGISTRY=https://registry.npmmirror.com
ARG PNPM_VERSION=10.33.0

FROM ${NODE_IMAGE} AS build

ARG PNPM_REGISTRY
ARG PNPM_VERSION
ENV PNPM_HOME="/pnpm"
ENV PATH="$PNPM_HOME:$PATH"
ENV COREPACK_NPM_REGISTRY=$PNPM_REGISTRY
ARG VITE_API_BASE_URL=
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL

WORKDIR /app/src/web

RUN npm config set registry "$PNPM_REGISTRY" \
  && corepack enable \
  && corepack prepare "pnpm@${PNPM_VERSION}" --activate \
  && pnpm config set registry "$PNPM_REGISTRY"

COPY src/web/package.json src/web/pnpm-lock.yaml ./
RUN pnpm install --frozen-lockfile

COPY src/web/ ./
RUN pnpm build

FROM ${NGINX_IMAGE} AS runtime

ENV API_UPSTREAM=http://backend:8000

COPY --from=build /app/src/web/dist /usr/share/nginx/html

EXPOSE 80

CMD ["/bin/sh", "-c", "cat > /etc/nginx/conf.d/default.conf <<EOF\nserver {\n  listen 80;\n  server_name _;\n  root /usr/share/nginx/html;\n  index index.html;\n\n  location /api/ {\n    proxy_pass ${API_UPSTREAM};\n    proxy_http_version 1.1;\n    proxy_set_header Host \\$host;\n    proxy_set_header X-Real-IP \\$remote_addr;\n    proxy_set_header X-Forwarded-For \\$proxy_add_x_forwarded_for;\n    proxy_set_header X-Forwarded-Proto \\$scheme;\n  }\n\n  location / {\n    try_files \\$uri \\$uri/ /index.html;\n  }\n}\nEOF\nnginx -g 'daemon off;'"]
