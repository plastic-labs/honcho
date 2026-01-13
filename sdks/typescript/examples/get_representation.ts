import { Honcho, type MessageCreate } from '../src';

/**
 * Example demonstrating how to get peer representations.
 *
 * This creates a session with random messages and retrieves both
 * global and local representations for a peer.
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
  const messages: MessageCreate[] = [];
  for (let i = 0; i < 10; i++) {
    const randomPeer = peers[Math.floor(Math.random() * peers.length)];
    messages.push(
      randomPeer.message(`Hello from ${randomPeer.id}! This is message ${i}.`)
    );
  }

  await session.addMessages(messages);
  console.log('Added 10 random messages to session.');

  const alice = peers[0];
  const bob = peers[1];

  console.log('Getting alice\'s working representation in session...');
  // Get alice's working representation in the session
  const representation = await session.getRepresentation(alice);
  console.log('Representation returned:', representation);

  console.log('Getting alice\'s working representation *of bob* in session...');
  // Get alice's working representation *of bob* in the session
  const representationOfBob = await session.getRepresentation(alice, bob);
  console.log('Representation returned:', representationOfBob);

  console.log('Example completed successfully!');
}

main().catch((err) => {
  console.error('Error running get_representation example:', err);
});
