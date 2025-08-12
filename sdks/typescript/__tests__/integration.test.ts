import { Honcho } from '../src/client';
import { Peer } from '../src/peer';
import { Session } from '../src/session';
import { SessionContext } from '../src/session_context';
import { Page } from '../src/pagination';

// Mock the @honcho-ai/core module
let mockWorkspacesApi: any;

jest.mock('@honcho-ai/core', () => {
  return jest.fn().mockImplementation(() => mockWorkspacesApi);
});

describe('Honcho SDK Integration Tests', () => {
  let honcho: Honcho;

  beforeEach(() => {
    mockWorkspacesApi = {
      workspaces: {
        peers: {
          list: jest.fn(),
          chat: jest.fn(),
          sessions: { list: jest.fn() },
          messages: { create: jest.fn(), list: jest.fn() },
          getOrCreate: jest.fn(),
          update: jest.fn(),
          search: jest.fn(),
          workingRepresentation: jest.fn(),
        },
        sessions: {
          list: jest.fn(),
          peers: { add: jest.fn(), set: jest.fn(), remove: jest.fn(), list: jest.fn() },
          messages: { create: jest.fn(), list: jest.fn() },
          getOrCreate: jest.fn(),
          update: jest.fn(),
          getContext: jest.fn(),
          search: jest.fn(),
        },
        getOrCreate: jest.fn(),
        update: jest.fn(),
        list: jest.fn(),
        search: jest.fn(),
      },
    };

    jest.clearAllMocks();

    honcho = new Honcho({
      workspaceId: 'integration-test-workspace',
      apiKey: 'test-api-key',
      environment: 'local',
    });
  });

  describe('Complete Workflow Integration', () => {
    it('should handle complete chat session workflow', async () => {
      // Setup mock responses
      const mockPeerData = { id: 'assistant', metadata: { role: 'ai' } };
      const mockSessionData = { id: 'chat-session', metadata: { topic: 'general' } };
      const mockMessages = [
        { id: 'msg1', content: 'Hello', peer_id: 'user' },
        { id: 'msg2', content: 'Hi there!', peer_id: 'assistant' },
      ];
      const mockContextData = {
        messages: mockMessages,
        summary: {
          content: 'Friendly greeting',
          message_id: 5,
          summary_type: 'short',
          created_at: '2024-01-01T00:00:00Z',
          token_count: 50
        }
      };

      mockWorkspacesApi.workspaces.peers.getOrCreate.mockResolvedValue(mockPeerData);
      mockWorkspacesApi.workspaces.sessions.getOrCreate.mockResolvedValue(mockSessionData);
      mockWorkspacesApi.workspaces.sessions.peers.add.mockResolvedValue({});
      mockWorkspacesApi.workspaces.sessions.messages.create.mockResolvedValue({});
      mockWorkspacesApi.workspaces.sessions.getContext.mockResolvedValue(mockContextData);
      mockWorkspacesApi.workspaces.peers.chat.mockResolvedValue({ content: 'AI response' });

      // Step 1: Create peers
      const user = await honcho.peer('user');
      const assistant = await honcho.peer('assistant');

      expect(user).toBeInstanceOf(Peer);
      expect(assistant).toBeInstanceOf(Peer);
      expect(user.id).toBe('user');
      expect(assistant.id).toBe('assistant');

      // Step 2: Create session
      const session = await honcho.session('chat-session');
      expect(session).toBeInstanceOf(Session);
      expect(session.id).toBe('chat-session');

      // Step 3: Add peers to session
      await session.addPeers([user, assistant]);
      expect(mockWorkspacesApi.workspaces.sessions.peers.add).toHaveBeenCalledWith(
        'integration-test-workspace',
        'chat-session',
        {
          'user': {},
          'assistant': {}
        }
      );

      // Step 4: Add messages to session
      const userMessage = user.message('Hello');
      const assistantMessage = assistant.message('Hi there!');

      await session.addMessages([userMessage, assistantMessage]);
      expect(mockWorkspacesApi.workspaces.sessions.messages.create).toHaveBeenCalledWith(
        'integration-test-workspace',
        'chat-session',
        {
          messages: [
            { peer_id: 'user', content: 'Hello', metadata: undefined },
            { peer_id: 'assistant', content: 'Hi there!', metadata: undefined },
          ]
        }
      );

      // Step 5: Get session context
      const context = await session.getContext();
      expect(context).toBeInstanceOf(SessionContext);
      expect(context.sessionId).toBe('chat-session');
      expect(context.messages).toEqual(mockMessages);
      expect(context.summary?.content).toBe('Friendly greeting');

      // Step 6: Convert context to different formats
      const openAIFormat = context.toOpenAI('assistant');
      const anthropicFormat = context.toAnthropic('assistant');

      expect(openAIFormat).toEqual([
        { role: 'system', content: '<summary>Friendly greeting</summary>' },
        { role: 'user', content: 'Hello', name: 'user' },
        { role: 'assistant', content: 'Hi there!', name: 'assistant' },
      ]);

      expect(anthropicFormat).toEqual([
        { role: 'user', content: '<summary>Friendly greeting</summary>' },
        { role: 'user', content: 'user: Hello' },
        { role: 'assistant', content: 'Hi there!' },
      ]);

      // Step 7: Query assistant
      const response = await assistant.chat('How are you?');
      expect(response).toBe('AI response');
      expect(mockWorkspacesApi.workspaces.peers.chat).toHaveBeenCalledWith(
        'integration-test-workspace',
        'assistant',
        { query: 'How are you?', stream: undefined, target: undefined, session_id: undefined }
      );
    });

    it('should handle workspace and peer management workflow', async () => {
      // Setup mock responses
      const mockWorkspaceMetadata = { name: 'Test Workspace', version: '1.0' };
      const mockPeersList = {
        items: [
          { id: 'peer1', metadata: { role: 'user' } },
          { id: 'peer2', metadata: { role: 'assistant' } },
        ],
        total: 2,
        size: 2,
        hasNextPage: false,
      };

      mockWorkspacesApi.workspaces.getOrCreate.mockResolvedValue({
        id: 'integration-test-workspace',
        metadata: mockWorkspaceMetadata,
      });
      mockWorkspacesApi.workspaces.update.mockResolvedValue({});
      mockWorkspacesApi.workspaces.peers.list.mockResolvedValue(mockPeersList);

      // Step 1: Get workspace metadata
      const metadata = await honcho.getMetadata();
      expect(metadata).toEqual(mockWorkspaceMetadata);

      // Step 2: Update workspace metadata
      const newMetadata = { ...mockWorkspaceMetadata, updated: true };
      await honcho.setMetadata(newMetadata);
      expect(mockWorkspacesApi.workspaces.update).toHaveBeenCalledWith(
        'integration-test-workspace',
        { metadata: newMetadata }
      );

      // Step 3: Get all peers
      const peersPage = await honcho.getPeers();
      expect(peersPage).toBeInstanceOf(Page);

      // Step 4: Iterate through peers
      const peersList: Peer[] = [];
      for await (const peer of peersPage) {
        peersList.push(peer);
      }

      expect(peersList).toHaveLength(2);
      expect(peersList[0]).toBeInstanceOf(Peer);
      expect(peersList[1]).toBeInstanceOf(Peer);
      expect(peersList[0].id).toBe('peer1');
      expect(peersList[1].id).toBe('peer2');
    });

    it('should handle search functionality across different scopes', async () => {
      // Setup mock responses
      const mockWorkspaceSearchResults = [
        { id: 'msg1', content: 'workspace message', peer_id: 'peer1' },
      ];

      const mockPeerSearchResults = [
        { id: 'msg2', content: 'peer message', peer_id: 'peer1' },
      ];

      const mockSessionSearchResults = [
        { id: 'msg3', content: 'session message', peer_id: 'peer1' },
      ];

      mockWorkspacesApi.workspaces.search.mockResolvedValue(mockWorkspaceSearchResults);
      mockWorkspacesApi.workspaces.peers.search.mockResolvedValue(mockPeerSearchResults);
      mockWorkspacesApi.workspaces.sessions.search.mockResolvedValue(mockSessionSearchResults);

      // Step 1: Search workspace
      const workspaceResults = await honcho.search('test query');
      expect(Array.isArray(workspaceResults)).toBe(true);
      expect(mockWorkspacesApi.workspaces.search).toHaveBeenCalledWith(
        'integration-test-workspace',
        { query: 'test query', limit: undefined }
      );

      // Step 2: Search peer
      const peer = await honcho.peer('test-peer');
      const peerResults = await peer.search('peer query');
      expect(Array.isArray(peerResults)).toBe(true);
      expect(mockWorkspacesApi.workspaces.peers.search).toHaveBeenCalledWith(
        'integration-test-workspace',
        'test-peer',
        { query: 'peer query', limit: undefined }
      );

      // Step 3: Search session
      const session = await honcho.session('test-session');
      const sessionResults = await session.search('session query');
      expect(Array.isArray(sessionResults)).toBe(true);
      expect(mockWorkspacesApi.workspaces.sessions.search).toHaveBeenCalledWith(
        'integration-test-workspace',
        'test-session',
        { query: 'session query', limit: undefined }
      );
    });

    it('should handle error scenarios gracefully', async () => {
      // Setup error scenarios
      mockWorkspacesApi.workspaces.peers.chat.mockRejectedValue(new Error('Chat API failed'));
      mockWorkspacesApi.workspaces.sessions.getContext.mockRejectedValue(new Error('Context API failed'));

      const assistant = await honcho.peer('assistant');
      const session = await honcho.session('error-session');

      // Test error handling in chat
      await expect(assistant.chat('Hello')).rejects.toThrow();

      // Test error handling in context
      await expect(session.getContext()).rejects.toThrow();
    });

    it('should handle pagination correctly', async () => {
      // Setup paginated response
      const firstPageData = {
        items: [
          { id: 'peer1', metadata: {} },
          { id: 'peer2', metadata: {} },
        ],
        total: 4,
        size: 2,
        hasNextPage: true,
        nextPage: jest.fn(),
      };

      const secondPageData = {
        items: [
          { id: 'peer3', metadata: {} },
          { id: 'peer4', metadata: {} },
        ],
        total: 4,
        size: 2,
        hasNextPage: false,
        nextPage: jest.fn().mockResolvedValue(null),
      };

      firstPageData.nextPage.mockResolvedValue(secondPageData);
      mockWorkspacesApi.workspaces.peers.list.mockResolvedValue(firstPageData);

      // Step 1: Get first page
      const firstPage = await honcho.getPeers();
      expect(firstPage.total).toBe(4);
      expect(firstPage.size).toBe(2);
      expect(firstPage.hasNextPage).toBe(true);

      // Step 2: Get data from first page
      const firstPageData_ = await firstPage.data();
      expect(firstPageData_).toHaveLength(2);

      // Step 3: Get next page
      const secondPage = await firstPage.nextPage();
      expect(secondPage).not.toBeNull();
      expect(secondPage!.hasNextPage).toBe(false);

      // Step 4: Get data from second page
      const secondPageData_ = await secondPage!.data();
      expect(secondPageData_).toHaveLength(2);

      // Step 5: Verify no more pages
      const thirdPage = await secondPage!.nextPage();
      expect(thirdPage).toBeNull();
    });

    it('should handle working representation queries', async () => {
      const mockWorkingRep = {
        peer_id: 'alice',
        knowledge: 'Alice likes coffee and works as a developer',
        relationships: ['bob', 'charlie'],
        context: 'session-specific context',
      };

      mockWorkspacesApi.workspaces.peers.workingRepresentation.mockResolvedValue(mockWorkingRep);

      const session = await honcho.session('working-rep-session');
      const alice = await honcho.peer('alice');
      const bob = await honcho.peer('bob');

      // Test working representation without target
      const globalRep = await session.workingRep('alice');
      expect(globalRep).toEqual(mockWorkingRep);
      expect(mockWorkspacesApi.workspaces.peers.workingRepresentation).toHaveBeenCalledWith(
        'integration-test-workspace',
        'alice',
        { session_id: 'working-rep-session', target: undefined }
      );

      // Test working representation with target
      await session.workingRep(alice, bob);
      expect(mockWorkspacesApi.workspaces.peers.workingRepresentation).toHaveBeenCalledWith(
        'integration-test-workspace',
        'alice',
        { session_id: 'working-rep-session', target: 'bob' }
      );
    });
  });

  describe('Edge Cases and Error Handling Integration', () => {
    it('should handle empty and null responses gracefully', async () => {
      // Setup empty/null responses
      mockWorkspacesApi.workspaces.peers.chat.mockResolvedValue({ content: null });
      mockWorkspacesApi.workspaces.peers.list.mockResolvedValue({ items: [], total: 0, hasNextPage: false });
      mockWorkspacesApi.workspaces.sessions.getContext.mockResolvedValue({ messages: [] });

      const peer = await honcho.peer('empty-peer');
      const session = await honcho.session('empty-session');

      // Test null chat response
      const chatResult = await peer.chat('Hello');
      expect(chatResult).toBeNull();

      // Test empty peers list
      const peersPage = await honcho.getPeers();
      const peersList = await peersPage.data();
      expect(peersList).toEqual([]);

      // Test empty context
      const context = await session.getContext();
      expect(context.messages).toEqual([]);
      expect(context.length).toBe(0);
    });

    it('should maintain type safety throughout the workflow', async () => {
      // This test verifies TypeScript types are maintained correctly
      const peer: Peer = await honcho.peer('typed-peer');
      const session: Session = await honcho.session('typed-session');

      expect(typeof peer.id).toBe('string');
      expect(typeof session.id).toBe('string');

      const message = peer.message('typed message', { metadata: { type: 'test' } });
      expect(typeof message.peer_id).toBe('string');
      expect(typeof message.content).toBe('string');
      expect(typeof message.metadata).toBe('object');

      // Mock successful operations
      mockWorkspacesApi.workspaces.sessions.getContext.mockResolvedValue({
        messages: [{ id: 'msg1', content: 'Hello', peer_id: 'typed-peer' }],
        summary: {
          content: 'Test summary',
          message_id: 1,
          summary_type: 'short',
          created_at: '2024-01-01T00:00:00Z',
          token_count: 20
        },
      });

      const context: SessionContext = await session.getContext();
      expect(typeof context.sessionId).toBe('string');
      expect(Array.isArray(context.messages)).toBe(true);
      expect(context.summary).not.toBeNull();
      expect(typeof context.summary?.content).toBe('string');
      expect(typeof context.length).toBe('number');
      expect(typeof context.toString()).toBe('string');

      const openAI = context.toOpenAI(peer);
      const anthropic = context.toAnthropic('assistant');

      expect(Array.isArray(openAI)).toBe(true);
      expect(Array.isArray(anthropic)).toBe(true);

      if (openAI.length > 0) {
        expect(typeof openAI[0].role).toBe('string');
        expect(typeof openAI[0].content).toBe('string');
      }
    });
  });
});
