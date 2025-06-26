import { Honcho, SessionPeerConfig } from '../src';

/**
 * Example usage of the Honcho TypeScript SDK.
 *
 * This demonstrates how to manage peers, sessions, and messages
 * using the high-level SDK API.
 */
async function main() {
  console.log('Initializing Honcho client...');
  const honcho = new Honcho({
    environment: 'local',
    workspaceId: 'test',
  });

  console.log('Creating peers...');
  const assistant = honcho.peer('bob');
  const alice = honcho.peer('alice');

  console.log('Fetching all peers in workspace...');
  const peers = await honcho.getPeers();
  for await (const peer of peers) {
    console.log('Peer:', peer.id);
  }

  console.log('Fetching workspace metadata...');
  const m = await honcho.getMetadata();
  console.log('Current metadata:', m);
  await honcho.setMetadata({ test: 'test' });
  console.log('Set workspace metadata.');

  console.log('Testing chat endpoint (should be null)...');
  const response = await alice.chat('what did alice have for breakfast today?');
  console.log('Chat response:', response);

  console.log('Creating session...');
  const mySession = honcho.session('session_1');

  console.log('Adding peers to session...');
  await mySession.addPeers([alice, [assistant, new SessionPeerConfig({ observe_me: false })]]);
  console.log('Peers added to session.');

  console.log('Fetching sessions for alice...');
  const _sessions = await alice.getSessions();
  for await (const session of _sessions) {
    console.log('Session:', session.id);
  }

  console.log('Adding messages to session...');
  await mySession.addMessages([
    assistant.message('what did you have for breakfast today, alice?'),
    alice.message('i had oatmeal.'),
  ]);
  console.log('Messages added.');

  let sessionMetadata = await mySession.getMetadata();
  console.log('Session metadata:', sessionMetadata);
  await mySession.setMetadata({ ...sessionMetadata, test: 'test2' });
  console.log('Session metadata updated.');

  console.log('Querying alice global representation...');
  await alice.chat('what did the user have for breakfast today?');

  console.log('Querying alice local representation of assistant...');
  await alice.chat('does alice know what bob had for breakfast?', { target: assistant });

  console.log('Querying assistant local representation of alice in session...');
  await assistant.chat('does the assistant know what alice had for breakfast?', {
    target: alice,
    sessionId: mySession.id,
  });

  console.log('Adding non-message content to alice...');
  await alice.addMessages('this might be a document about alice, say, a journal entry.');

  console.log('Creating charlie peer and adding message...');
  const charlie = honcho.peer('charlie');
  await mySession.addMessages(charlie.message('hello world!'));

  console.log('Fetching and updating charlie metadata...');
  let charlieMetadata = await charlie.getMetadata();
  await charlie.setMetadata({ ...charlieMetadata, location: 'the moon' });
  console.log('Charlie metadata updated.');

  console.log('Querying charlie for location...');
  await charlie.chat('where is the user?');

  console.log('Fetching all messages from session...');
  const messages = await mySession.getMessages();
  console.log('Messages:', messages.total);

  console.log('Fetching session context...');
  const context = await mySession.getContext();
  let openaiMessages = context.toOpenAI(alice.id);
  let anthropicMessages = context.toAnthropic(alice.id);
  console.log('OpenAI context:', openaiMessages);
  console.log('Anthropic context:', anthropicMessages);

  console.log('Adding test message using property syntax...');
  await mySession.addMessages(
    assistant.message('This is a test message using the property syntax')
  );

  console.log('Sample code executed successfully!');
}

main().catch((err) => {
  console.error('Error running example:', err);
}); 