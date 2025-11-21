/**
 * LangGraph Integration with Honcho and OpenAI
 *
 * This module demonstrates how to build a stateful conversational AI agent using
 * LangGraph for orchestration, OpenAI for the AI model, and Honcho for memory
 * management. It creates a chatbot that remembers conversations across sessions.
 */

import * as dotenv from "dotenv";
import { Honcho, Peer, Session } from "@honcho-ai/sdk";
import OpenAI from "openai";
import { Annotation } from "@langchain/langgraph";
import { StateGraph, START, END } from "@langchain/langgraph";
import * as readline from "readline/promises";

dotenv.config();

const honcho = new Honcho({});

const llm = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY
});

const StateAnnotation = Annotation.Root({
  userMessage: Annotation<string>(),
  assistantResponse: Annotation<string>(),
  user: Annotation<Peer>(),
  assistant: Annotation<Peer>(),
  session: Annotation<Session>(),
});

type State = typeof StateAnnotation.State;

async function chatbot(state: State) {
  const userMessage = state.userMessage;

  // Get objects from state
  const user = state.user;
  const assistant = state.assistant;
  const session = state.session;
  await session.addMessages([user.message(userMessage)]);

  // Get context in OpenAI format with token limit
  // tokens: 2000 limits the context to 2000 tokens to manage costs and fit within model limits
  const messages = (await session.getContext({ tokens: 2000 })).toOpenAI(assistant);

  // Generate response
  const response = await llm.chat.completions.create({
    model: "gpt-4o",
    messages: messages
  });
  const assistantResponse = response.choices[0].message.content!;

  // Store assistant response
  await session.addMessages([assistant.message(assistantResponse)]);

  return { assistantResponse: assistantResponse };
}

const graph = new StateGraph(StateAnnotation)
  .addNode("chatbot", chatbot)
  .addEdge(START, "chatbot")
  .addEdge("chatbot", END)
  .compile();

async function runConversationTurn(
  userId: string,
  userInput: string,
  sessionId?: string
): Promise<string> {
  if (!sessionId) {
    sessionId = `session_${userId}`;
  }

  // Initialize Honcho objects
  const user = await honcho.peer(userId);
  const assistant = await honcho.peer("assistant");
  const session = await honcho.session(sessionId);

  const result = await graph.invoke({
    userMessage: userInput,
    user: user,
    assistant: assistant,
    session: session,
  });

  return result.assistantResponse;
}

async function main() {
  console.log("Welcome to the AI Assistant! How can I help you today?");
  const userId = "test-user-1234";

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  while (true) {
    const userInput = await rl.question("You: ");
    if (userInput.toLowerCase() === "quit" || userInput.toLowerCase() === "exit") {
      rl.close();
      break;
    }
    const response = await runConversationTurn(userId, userInput);
    console.log(`Assistant: ${response}\n`);
  }
}

main();
