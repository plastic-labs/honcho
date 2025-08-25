import { Honcho, Message } from '../src';

/**
 * Example demonstrating how to get context from a session with summary and token limits.
 *
 * This creates a session with random messages and retrieves context
 * with a low token limit to demonstrate the summarization feature.
 */
async function main() {
  console.log('Creating Honcho client...');
  // Create a Honcho client with the default workspace
  const honcho = new Honcho({
    environment: 'local'
  });

  console.log('Creating peers...');
  const peers = [
    await honcho.peer('alice'),
    await honcho.peer('bob'),
    await honcho.peer('charlie'),
  ];

  // Create a new session
  const sessionId = `context_test_${crypto.randomUUID()}`;
  const session = await honcho.session(sessionId);
  console.log(`Created session: ${sessionId}`);

  console.log('Generating random messages...');
  // Generate some random messages from alice, bob, and charlie and add them to the session
  const messages: Message[] = [];
  for (let i = 0; i < 10; i++) {
    const randomPeer = peers[Math.floor(Math.random() * peers.length)];
    messages.push(
      randomPeer.message(`Hello from ${randomPeer.id}! This is message ${i}.`)
    );
  }

  await session.addMessages(messages);
  console.log('Added 10 random messages to session.');

  console.log('Getting context with summary and low token limit...');
  // Get some context of the session
  // Set the token limit super low so we only get a few of the tiny messages created
  const context = await session.getContext({ summary: true, tokens: 50 });
  console.log('Context returned:', context);

  console.log('Example completed successfully!');
}

main().catch((err) => {
  console.error('Error running get_context example:', err);
});
