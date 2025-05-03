import { genkit, z } from 'genkit/beta';
import { ai } from "./genkit"

export const getEvents = ai.defineTool(
    {
        name: "eventsTool",
        description: 'Gets the current events in a given location',
        inputSchema: z.object({ 
          location: z.string().describe('The location to get the current events for')
        }),
        outputSchema: z.string()
    },
    async (input, {context, interrupt, resumed}) => {
        console.log('Input:', input);
        return "List of events in " + input.location + ":\n Event 1\n, Event 2\n, Event 3.";
    }
);