# Build stage
FROM golang:1.25.6-bookworm AS builder

WORKDIR /app

# Copy go mod files
COPY go.mod go.sum ./

# Download dependencies
RUN go mod download

# Copy source code
COPY . .

# Build the binary
RUN go build -o copilot-server .

# Runtime stage
FROM node:22-bookworm

# Install GitHub Copilot CLI
RUN npm install -g @github/copilot

# Copy the built binary from builder stage
COPY --from=builder /app/copilot-server /usr/local/bin/copilot-server

# Expose port
EXPOSE 8080

# Run the server
CMD ["copilot-server"]
