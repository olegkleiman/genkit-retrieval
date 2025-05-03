import {
    A2AServer,
    TaskContext,
    // TaskYieldUpdate,
    // schema,
  } from "../server/server.js";
import { TaskYieldUpdate } from "../server/handler.js";
 
import { ai } from "../genkit.js"
import { getEvents } from '../tools.js';
import { AgentCard } from "../schema.js";

// This is Dotprompt (see https://github.com/google/dotprompt)
const prompt = ai.prompt('tools_agent'); // '.prompt' extension will be added automatically

// --- Server Setup ---

const paymentsAgentCard: AgentCard = {
    name: "Paymemnts Agent",
    description: "An agent that can perform payments on your behalf.",
    url: "http://localhost:41241", // Default port used in the script
    provider: {
        organization: "Credit Gard",
    },
    version: "0.0.1",
    capabilities: {
        // Although it yields multiple updates, it doesn't seem to implement full A2A streaming via TaskYieldUpdate artifacts
        // It uses Genkit streaming internally, but the A2A interface yields start/end messages.
        // State history seems reasonable as it processes history.
        streaming: false,
        pushNotifications: false,
        stateTransitionHistory: true,
    },
    authentication: null,
    skills: [
        {
            id: "general_payments_system",
            name: "Payments Agent",
            tags: ["payments", "credit card", "banking"],
        }]
}

async function* paymentsHelper(
    context: TaskContext
    ): AsyncGenerator<TaskYieldUpdate>{
    yield {
        state: 'working',
        message: {
            role: "agent",
            parts: [{ type:"text", text: "Processing your question, hang tight!" }],
        },        
    }

    try {
        const prompt = ai.prompt('tools_agent'); // '.prompt' extension will be added automatically
        const promptResponse = await prompt({ // Full conversation turn with LLM here. After it completes, 'text' contains the response text
            // Prompt input
            input,
            docs: docs
        },
        {
            tools: [getEvents]
        });
        const responseText = promptResponse.text; // Access the text property directly
        console.log("Response to prompt: ", responseText);

              // Yield the final result
        yield {
            state: finalState,
            message: {
            role: "agent",
            parts: [{ type: "text", text: agentReply }],
            },
        };

    } catch (error: any) {
      // Yield a failed state if the prompt execution fails
      yield {
        state: "failed",
        message: {
          role: "agent",
          parts: [{ type: "text", text: `Agent error: ${error.message}` }],
        },
      };
    }

}

// Create server with the task handler. Defaults to InMemoryTaskStore.
const server = new A2AServer(paymentsHelper, { card: paymentsAgentCard });