# Use the official Node.js 22 image
FROM node:22.16.0-slim

# Set working directory
WORKDIR /app

# Install the inspector package globally (optional, see below)
RUN npm install -g @modelcontextprotocol/inspector
EXPOSE 6274 6277
# Default command runs the inspector via npx
CMD ["npx", "@modelcontextprotocol/inspector"]
