import { Honcho, Message } from '../src';

/**
 * Example demonstrating search functionality across different scopes.
 *
 * This creates sessions with special keywords and demonstrates
 * searching at session, workspace, and peer levels.
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

  const alice = peers[0];

  // Create a new session
  const sessionId = `search_test_${crypto.randomUUID()}`;
  const session = await honcho.session(sessionId);
  console.log(`Created session: ${sessionId}`);

  // Create a message with our special keyword
  const keyword = `~special-${crypto.randomUUID()}~`;
  console.log(`Using keyword: ${keyword}`);
  await session.addMessages(alice.message(`I am a ${keyword} message`));

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
  console.log('Added random messages to session.');

  console.log('Searching the session...');
  // Search the session for the special keyword
  const sessionSearchResults = await session.search(keyword);
  console.log(`Session search returned ${sessionSearchResults.length} results:`);
  for (const message of sessionSearchResults) {
    console.log(`  - ${message.content} (from ${message.peer_id})`);
  }

  console.log('Searching the workspace...');
  // Search the workspace for the special keyword
  const workspaceSearchResults = await honcho.search(keyword);
  console.log(`Workspace search returned ${workspaceSearchResults.length} results:`);
  for (const message of workspaceSearchResults) {
    console.log(`  - ${message.content} (from ${message.peer_id})`);
  }

  console.log('Searching alice\'s messages...');
  // Search alice's messages for the special keyword
  const aliceSearchResults = await alice.search(keyword);
  console.log(`Alice search returned ${aliceSearchResults.length} results:`);
  for (const message of aliceSearchResults) {
    console.log(`  - ${message.content} (from ${message.peer_id})`);
  }

  console.log('Example completed successfully!');
}

main().catch((err) => {
  console.error('Error running search example:', err);
});
