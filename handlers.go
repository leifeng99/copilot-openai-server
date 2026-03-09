package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strings"
	"sync"
	"time"

	copilot "github.com/github/copilot-sdk/go"
)

// Server holds the copilot client and configuration
type Server struct {
	client *copilot.Client
	mu     sync.Mutex
}

// NewServer creates a new server instance
func NewServer() (*Server, error) {
	client := copilot.NewClient(&copilot.ClientOptions{
		LogLevel: "error",
	})

	if err := client.Start(context.Background()); err != nil {
		return nil, fmt.Errorf("failed to start copilot client: %w", err)
	}

	return &Server{client: client}, nil
}

// Close stops the copilot client
func (s *Server) Close() {
	s.client.Stop()
}

// HandleModels handles GET /v1/models
func (s *Server) HandleModels(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "Method not allowed", "invalid_request_error")
		return
	}

	models, err := s.client.ListModels(r.Context())
	if err != nil {
		log.Printf("Error listing models: %v", err)
		writeError(w, http.StatusInternalServerError, "Failed to list models", "api_error")
		return
	}

	response := ModelsResponse{
		Object: "list",
		Data:   make([]ModelData, 0, len(models)),
	}

	for _, model := range models {
		response.Data = append(response.Data, ModelData{
			ID:      model.ID,
			Object:  "model",
			Created: currentTimestamp(),
			OwnedBy: "github-copilot",
		})
	}

	writeJSON(w, http.StatusOK, response)
}

// HandleChatCompletions handles POST /v1/chat/completions
func (s *Server) HandleChatCompletions(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "Method not allowed", "invalid_request_error")
		return
	}

	var req ChatCompletionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "Invalid request body", "invalid_request_error")
		return
	}

	if req.Model == "" {
		writeError(w, http.StatusBadRequest, "Model is required", "invalid_request_error")
		return
	}

	if len(req.Messages) == 0 {
		writeError(w, http.StatusBadRequest, "Messages are required", "invalid_request_error")
		return
	}

	// Extract system message - iterate through all messages to find system/developer roles
	var systemMessageParts []string
	for _, msg := range req.Messages {
		if msg.Role == "system" || msg.Role == "developer" {
			systemMessageParts = append(systemMessageParts, msg.Content)
		}
	}

	// Build the prompt from messages (excluding system messages which are handled separately)
	prompt := buildPrompt(req.Messages)

	// Convert OpenAI tools to Copilot tools (definitions only, no handlers)
	var copilotTools []copilot.Tool
	log.Printf("[DEBUG] Received %d tools in request", len(req.Tools))
	for _, tool := range req.Tools {
		if tool.Type == "function" {
			// toolJSON, _ := json.MarshalIndent(tool, "", "  ")
			// log.Printf("[DEBUG] Tool %d: %s", i, string(toolJSON))
			copilotTools = append(copilotTools, copilot.Tool{
				Name:        tool.Function.Name,
				Description: tool.Function.Description,
				Parameters:  tool.Function.Parameters,
				// No handler - we just want to capture tool calls
			})
		}
	}

	// Create session config
	sessionConfig := &copilot.SessionConfig{
		Model:     req.Model,
		Streaming: req.Stream,
		Tools:     copilotTools,
		// Disable infinite sessions for simple request/response
		InfiniteSessions: &copilot.InfiniteSessionConfig{
			Enabled: copilot.Bool(false),
		},
	}

	// Add system message if present
	if len(systemMessageParts) > 0 {
		systemContent := strings.Join(systemMessageParts, "\n\n")
		log.Printf("[DEBUG] Setting system message (length: %d)", len(systemContent))
		sessionConfig.SystemMessage = &copilot.SystemMessageConfig{
			Mode:    "replace",
			Content: systemContent,
		}
	}

	// If tools are provided, we want to limit available tools to only our custom ones
	// This prevents Copilot from using built-in file/git tools
	if len(copilotTools) > 0 {
		toolNames := make([]string, len(copilotTools))
		for i, t := range copilotTools {
			toolNames[i] = t.Name
		}
		sessionConfig.AvailableTools = toolNames
	}

	// Log session config
	// log.Printf("[DEBUG] Creating session with model: %s, streaming: %v, tools: %d",
	// 	req.Model, req.Stream, len(copilotTools))
	for i, t := range copilotTools {
		_ = i
		_ = t
		// paramsJSON, _ := json.Marshal(t.Parameters)
		// log.Printf("[DEBUG] Copilot Tool %d: name=%s, desc=%s, params=%s",
		// 	i, t.Name, t.Description, string(paramsJSON))
	}

	// Create session
	session, err := s.client.CreateSession(r.Context(), sessionConfig)
	if err != nil {
		log.Printf("[ERROR] Creating session failed: %v", err)
		writeError(w, http.StatusInternalServerError, "Failed to create session", "api_error")
		return
	}
	defer session.Destroy()
	log.Printf("[DEBUG] Session created successfully")

	// Log the full prompt being sent
	// log.Printf("[DEBUG] Full prompt being sent:\n%s", prompt)

	if req.Stream {
		log.Printf("[DEBUG] Starting streaming response")
		s.handleStreamingResponse(r.Context(), w, session, prompt, req.Model)
	} else {
		log.Printf("[DEBUG] Starting non-streaming response")
		s.handleNonStreamingResponse(r.Context(), w, session, prompt, req.Model)
	}
}

// handleNonStreamingResponse handles non-streaming chat completions
func (s *Server) handleNonStreamingResponse(ctx context.Context, w http.ResponseWriter, session *copilot.Session, prompt, model string) {
	var contentBuilder strings.Builder
	var toolCalls []ToolCall
	var finishReason string = "stop"

	done := make(chan bool)
	var closeOnce sync.Once

	session.On(func(event copilot.SessionEvent) {
		switch event.Type {
		case copilot.AssistantMessage:
			// Check for tool requests
			if len(event.Data.ToolRequests) > 0 {
				finishReason = "tool_calls"
				for _, tr := range event.Data.ToolRequests {
					argsJSON, _ := json.Marshal(tr.Arguments)
					toolCalls = append(toolCalls, ToolCall{
						ID:   tr.ToolCallID,
						Type: "function",
						Function: ToolCallFunction{
							Name:      tr.Name,
							Arguments: string(argsJSON),
						},
					})
				}
			}
			// Capture final content
			if event.Data.Content != nil {
				contentBuilder.WriteString(*event.Data.Content)
			}

		case copilot.SessionIdle:
			closeOnce.Do(func() { close(done) })

		case copilot.SessionError:
			if event.Data.Message != nil {
				log.Printf("Session error: %s", *event.Data.Message)
			}
			closeOnce.Do(func() { close(done) })
		}
	})

	// Send the message
	_, err := session.Send(ctx, copilot.MessageOptions{
		Prompt: prompt,
	})
	if err != nil {
		log.Printf("Error sending message: %v", err)
		writeError(w, http.StatusInternalServerError, "Failed to send message", "api_error")
		return
	}

	// Wait for completion with timeout
	select {
	case <-done:
	case <-ctx.Done():
		log.Printf("Request canceled: %v", ctx.Err())
		return
	case <-time.After(5 * time.Minute):
		log.Printf("Request timed out")
		writeError(w, http.StatusGatewayTimeout, "Request timed out", "api_error")
		return
	}

	// Build response
	response := ChatCompletionResponse{
		ID:      fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano()),
		Object:  "chat.completion",
		Created: currentTimestamp(),
		Model:   model,
		Choices: []Choice{
			{
				Index: 0,
				Message: &Message{
					Role:      "assistant",
					Content:   contentBuilder.String(),
					ToolCalls: toolCalls,
				},
				FinishReason: &finishReason,
			},
		},
	}

	writeJSON(w, http.StatusOK, response)
}

// handleStreamingResponse handles streaming chat completions with SSE
func (s *Server) handleStreamingResponse(ctx context.Context, w http.ResponseWriter, session *copilot.Session, prompt, model string) {
	// Set SSE headers
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")

	flusher, ok := w.(http.Flusher)
	if !ok {
		writeError(w, http.StatusInternalServerError, "Streaming not supported", "api_error")
		return
	}

	completionID := fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano())
	done := make(chan bool)
	var toolCalls []ToolCall
	var mu sync.Mutex

	sendChunk := func(delta Message, finishReason *string) {
		chunk := ChatCompletionChunk{
			ID:      completionID,
			Object:  "chat.completion.chunk",
			Created: currentTimestamp(),
			Model:   model,
			Choices: []Choice{
				{
					Index:        0,
					Delta:        &delta,
					FinishReason: finishReason,
				},
			},
		}
		data, _ := json.Marshal(chunk)
		if finishReason != nil {
			log.Printf("[DEBUG] SSE chunk (finish=%s): %s", *finishReason, string(data))
		}
		fmt.Fprintf(w, "data: %s\n\n", data)
		flusher.Flush()
	}

	// Send initial chunk with role
	sendChunk(Message{Role: "assistant"}, nil)

	var closeOnce sync.Once
	session.On(func(event copilot.SessionEvent) {
		// log.Printf("[DEBUG] Received event: %s", event.Type)
		switch event.Type {
		case copilot.AssistantMessageDelta:
			// Stream content deltas
			if event.Data.DeltaContent != nil {
				sendChunk(Message{Content: *event.Data.DeltaContent}, nil)
			}

		case copilot.AssistantMessage:
			log.Printf("[DEBUG] AssistantMessage - ToolRequests: %d, Content length: %d",
				len(event.Data.ToolRequests),
				func() int {
					if event.Data.Content != nil {
						return len(*event.Data.Content)
					}
					return 0
				}())
			// Check for tool requests
			if len(event.Data.ToolRequests) > 0 {
				log.Printf("[DEBUG] Tool calls found - streaming to client incrementally")
				mu.Lock()
				for i, tr := range event.Data.ToolRequests {
					argsJSON, _ := json.Marshal(tr.Arguments)
					// log.Printf("[DEBUG]   Tool %d: %s with args: %s", i, tr.Name, string(argsJSON))
					idx := i
					// Store for final chunk
					toolCalls = append(toolCalls, ToolCall{
						Index: &idx,
						ID:    tr.ToolCallID,
						Type:  "function",
						Function: ToolCallFunction{
							Name:      tr.Name,
							Arguments: string(argsJSON),
						},
					})
					// Stream tool call incrementally: first send id/type/name
					sendChunk(Message{ToolCalls: []ToolCall{{
						Index: &idx,
						ID:    tr.ToolCallID,
						Type:  "function",
						Function: ToolCallFunction{
							Name: tr.Name,
						},
					}}}, nil)
					// Then send arguments
					sendChunk(Message{ToolCalls: []ToolCall{{
						Index: &idx,
						Function: ToolCallFunction{
							Arguments: string(argsJSON),
						},
					}}}, nil)
				}
				mu.Unlock()
				// Return immediately - client needs to execute tools and send results back
				closeOnce.Do(func() { close(done) })
			}

		case copilot.SessionIdle:
			log.Printf("[DEBUG] SessionIdle - completing request")
			closeOnce.Do(func() { close(done) })

		case copilot.SessionError:
			if event.Data.Message != nil {
				log.Printf("[DEBUG] SessionError: %s", *event.Data.Message)
			}
			closeOnce.Do(func() { close(done) })

		default:
			// log.Printf("[DEBUG] Unhandled event type: %s", event.Type)
		}
	})

	// Send the message
	_, err := session.Send(ctx, copilot.MessageOptions{
		Prompt: prompt,
	})
	if err != nil {
		log.Printf("Error sending message: %v", err)
		return
	}

	// Wait for completion
	select {
	case <-done:
	case <-ctx.Done():
		log.Printf("Streaming request canceled: %v", ctx.Err())
		return
	case <-time.After(5 * time.Minute):
		log.Printf("Streaming request timed out")
		return
	}

	// Send final chunk with finish_reason
	mu.Lock()
	if len(toolCalls) > 0 {
		// log.Printf("[DEBUG] Tool calls already streamed, sending finish_reason only")
		// Don't resend tool calls - they were already streamed incrementally
		sendChunk(Message{}, strPtr("tool_calls"))
	} else {
		sendChunk(Message{}, strPtr("stop"))
	}
	mu.Unlock()

	// Send [DONE]
	// log.Printf("[DEBUG] Sending [DONE] marker")
	fmt.Fprintf(w, "data: [DONE]\n\n")
	flusher.Flush()
}

// buildPrompt converts OpenAI messages to a single prompt string
func buildPrompt(messages []Message) string {
	var parts []string

	for _, msg := range messages {
		switch msg.Role {

		case "system", "developer":
			// System/developer messages are handled via SessionConfig.SystemMessage
			continue
		case "user":
			parts = append(parts, fmt.Sprintf("[User]: %s", msg.Content))
		case "assistant":
			if msg.Content != "" {
				parts = append(parts, fmt.Sprintf("[Assistant]: %s", msg.Content))
			}
			// Handle previous tool calls
			for _, tc := range msg.ToolCalls {
				parts = append(parts, fmt.Sprintf("[Assistant called tool %s with args: %s]", tc.Function.Name, tc.Function.Arguments))
			}
		case "tool":
			parts = append(parts, fmt.Sprintf("[Tool result for %s]: %s", msg.ToolCallID, msg.Content))
		}
	}

	return strings.Join(parts, "\n\n")
}

// writeJSON writes a JSON response
func writeJSON(w http.ResponseWriter, status int, data interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(data)
}

// writeError writes an error response
func writeError(w http.ResponseWriter, status int, message, errType string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(ErrorResponse{
		Error: ErrorDetail{
			Message: message,
			Type:    errType,
		},
	})
}
