# 前端
docker build \
  -f docker/frontend.dockerfile \
  -t registry.cn-hangzhou.aliyuncs.com/migo-dl/prama-server-frontend:0.1.0 \
  .

# 后端
docker build \
  -f docker/backend.dockerfile \
  -t registry.cn-hangzhou.aliyuncs.com/migo-dl/prama-server-backend:0.1.0 \
  .