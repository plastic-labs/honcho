import { Peer } from '../src/peer';
import { Session } from '../src/session';
import { Page } from '../src/pagination';
import { Honcho } from '../src/client';
import { Representation } from '../src/representation';

// Mock the @honcho-ai/core module
jest.mock('@honcho-ai/core', () => {
  return jest.fn().mockImplementation(() => ({
    workspaces: {
      peers: {
        chat: jest.fn(),
        sessions: {
          list: jest.fn(),
        },
        messages: {
          create: jest.fn(),
          list: jest.fn(),
        },
        getOrCreate: jest.fn(),
        update: jest.fn(),
        search: jest.fn(),
      },
      getOrCreate: jest.fn().mockResolvedValue({ id: 'test-workspace', metadata: {} }),
      update: jest.fn(),
      list: jest.fn(),
      search: jest.fn(),
    },
  }));
});

describe('Peer', () => {
  let honcho: Honcho;
  let peer: Peer;
  let mockClient: any;

  beforeEach(() => {
    jest.clearAllMocks();

    honcho = new Honcho({
      workspaceId: 'test-workspace',
      apiKey: 'test-key',
      environment: 'local',
    });

    peer = new Peer('test-peer', 'test-workspace', (honcho as any)._client);
    mockClient = (honcho as any)._client;
  });

  describe('constructor', () => {
    it('should initialize with correct properties', () => {
      const newPeer = new Peer('peer-id', 'test-workspace', mockClient);

      expect(newPeer.id).toBe('peer-id');
      expect(newPeer.workspaceId).toBe('test-workspace');
      expect(newPeer['_client']).toBe(mockClient);
    });
  });

  describe('chat', () => {
    it('should query peer representation and return response', async () => {
      const mockResponse = { content: 'Hello, I am a peer response' };
      mockClient.workspaces.peers.chat.mockResolvedValue(mockResponse);

      const result = await peer.chat('Hello');

      expect(result).toBe('Hello, I am a peer response');
      expect(mockClient.workspaces.peers.chat).toHaveBeenCalledWith(
        'test-workspace',
        'test-peer',
        { query: 'Hello', stream: false, target: undefined, session_id: undefined }
      );
    });

    it('should return null for None content', async () => {
      const mockResponse = { content: 'None' };
      mockClient.workspaces.peers.chat.mockResolvedValue(mockResponse);

      const result = await peer.chat('Hello');

      expect(result).toBeNull();
    });

    it('should return null for empty content', async () => {
      const mockResponse = { content: null };
      mockClient.workspaces.peers.chat.mockResolvedValue(mockResponse);

      const result = await peer.chat('Hello');

      expect(result).toBeNull();
    });

    it.skip('should handle chat with streaming option', async () => {
      // Skipped: streaming now uses fetch API directly, not the mocked client
      // TODO: Add proper streaming tests with fetch mocking when needed
    });

    it('should handle chat with target peer', async () => {
      const targetPeer = new Peer('target-peer', 'test-workspace', mockClient);
      const mockResponse = { content: 'Targeted response' };
      mockClient.workspaces.peers.chat.mockResolvedValue(mockResponse);

      await peer.chat('Hello', { target: targetPeer });

      expect(mockClient.workspaces.peers.chat).toHaveBeenCalledWith(
        'test-workspace',
        'test-peer',
        { query: 'Hello', stream: false, target: 'target-peer', session_id: undefined }
      );
    });

    it('should handle chat with target as string', async () => {
      const mockResponse = { content: 'Targeted response' };
      mockClient.workspaces.peers.chat.mockResolvedValue(mockResponse);

      await peer.chat('Hello', { target: 'string-target' });

      expect(mockClient.workspaces.peers.chat).toHaveBeenCalledWith(
        'test-workspace',
        'test-peer',
        { query: 'Hello', stream: false, target: 'string-target', session_id: undefined }
      );
    });

    it('should handle chat with session ID', async () => {
      const mockResponse = { content: 'Session-specific response' };
      mockClient.workspaces.peers.chat.mockResolvedValue(mockResponse);

      await peer.chat('Hello', { sessionId: 'session-123' });

      expect(mockClient.workspaces.peers.chat).toHaveBeenCalledWith(
        'test-workspace',
        'test-peer',
        { query: 'Hello', stream: false, target: undefined, session_id: 'session-123' }
      );
    });

    // TODO: Re-enable after regenerating Stainless SDK with streaming support
    // it('should handle all options together', async () => {
    //   const targetPeer = new Peer('target-peer', 'test-workspace', mockClient);
    //   const mockResponse = { content: 'Full options response' };
    //   mockClient.workspaces.peers.chat.mockResolvedValue(mockResponse);

    //   await peer.chat('Hello', { stream: true, target: targetPeer, sessionId: 'session-456' });

    //   expect(mockClient.workspaces.peers.chat).toHaveBeenCalledWith(
    //     'test-workspace',
    //     'test-peer',
    //     { query: 'Hello', stream: true, target: 'target-peer', session_id: 'session-456' }
    //   );
    // });

    it('should handle API errors', async () => {
      mockClient.workspaces.peers.chat.mockRejectedValue(new Error('Chat failed'));

      await expect(peer.chat('Hello')).rejects.toThrow();
    });
  });

  describe('getSessions', () => {
    it('should return Page of Session instances', async () => {
      const mockSessionsData = {
        items: [
          { id: 'session1', metadata: {} },
          { id: 'session2', metadata: {} },
        ],
        total: 2,
        size: 2,
        hasNextPage: false,
      };
      mockClient.workspaces.peers.sessions.list.mockResolvedValue(mockSessionsData);

      const sessionsPage = await peer.getSessions();

      expect(sessionsPage).toBeInstanceOf(Page);
      expect(mockClient.workspaces.peers.sessions.list).toHaveBeenCalledWith(
        'test-workspace',
        'test-peer',
        { filters: undefined }
      );
    });

    it('should handle empty sessions list', async () => {
      const mockSessionsData = {
        items: [],
        total: 0,
        size: 0,
        hasNextPage: false,
      };
      mockClient.workspaces.peers.sessions.list.mockResolvedValue(mockSessionsData);

      const sessionsPage = await peer.getSessions();

      expect(sessionsPage).toBeInstanceOf(Page);
    });

    it('should handle API errors', async () => {
      mockClient.workspaces.peers.sessions.list.mockRejectedValue(new Error('Failed to get sessions'));

      await expect(peer.getSessions()).rejects.toThrow();
    });
  });

  describe('message', () => {
    it('should create message object without metadata', () => {
      const message = peer.message('Test content');

      expect(message).toEqual({
        peer_id: 'test-peer',
        content: 'Test content',
        metadata: undefined,
      });
    });

    it('should create message object with metadata', () => {
      const metadata = { importance: 'high', category: 'greeting' };
      const message = peer.message('Hello there', { metadata });

      expect(message).toEqual({
        peer_id: 'test-peer',
        content: 'Hello there',
        metadata: { importance: 'high', category: 'greeting' },
      });
    });

    it('should handle empty content', () => {
      const message = peer.message('');

      expect(message).toEqual({
        peer_id: 'test-peer',
        content: '',
        metadata: undefined,
      });
    });
  });

  describe('getMetadata', () => {
    it('should return peer metadata', async () => {
      const mockPeer = {
        id: 'test-peer',
        metadata: { name: 'Test Peer', role: 'assistant' },
      };
      mockClient.workspaces.peers.getOrCreate.mockResolvedValue(mockPeer);

      const metadata = await peer.getMetadata();

      expect(metadata).toEqual({ name: 'Test Peer', role: 'assistant' });
      expect(mockClient.workspaces.peers.getOrCreate).toHaveBeenCalledWith(
        'test-workspace',
        { id: 'test-peer' }
      );
    });

    it('should return empty object when no metadata exists', async () => {
      const mockPeer = {
        id: 'test-peer',
        metadata: null,
      };
      mockClient.workspaces.peers.getOrCreate.mockResolvedValue(mockPeer);

      const metadata = await peer.getMetadata();

      expect(metadata).toEqual({});
    });

    it('should handle API errors', async () => {
      mockClient.workspaces.peers.getOrCreate.mockRejectedValue(new Error('Peer not found'));

      await expect(peer.getMetadata()).rejects.toThrow();
    });
  });

  describe('setMetadata', () => {
    it('should update peer metadata', async () => {
      const metadata = { name: 'Updated Peer', status: 'active' };
      mockClient.workspaces.peers.update.mockResolvedValue({});

      await peer.setMetadata(metadata);

      expect(mockClient.workspaces.peers.update).toHaveBeenCalledWith(
        'test-workspace',
        'test-peer',
        { metadata }
      );
    });

    it('should handle empty metadata', async () => {
      mockClient.workspaces.peers.update.mockResolvedValue({});

      await peer.setMetadata({});

      expect(mockClient.workspaces.peers.update).toHaveBeenCalledWith(
        'test-workspace',
        'test-peer',
        { metadata: {} }
      );
    });

    it('should handle complex metadata objects', async () => {
      const complexMetadata = {
        profile: { name: 'Complex Peer', age: 25 },
        settings: { theme: 'dark', notifications: true },
        tags: ['ai', 'assistant', 'helpful'],
      };
      mockClient.workspaces.peers.update.mockResolvedValue({});

      await peer.setMetadata(complexMetadata);

      expect(mockClient.workspaces.peers.update).toHaveBeenCalledWith(
        'test-workspace',
        'test-peer',
        { metadata: complexMetadata }
      );
    });

    it('should handle API errors', async () => {
      mockClient.workspaces.peers.update.mockRejectedValue(new Error('Update failed'));

      await expect(peer.setMetadata({ key: 'value' })).rejects.toThrow();
    });
  });

  describe('getPeerConfig', () => {
    it('should return peer configuration', async () => {
      const mockPeer = {
        id: 'test-peer',
        configuration: { observe_me: true, observe_others: false },
      };
      mockClient.workspaces.peers.getOrCreate.mockResolvedValue(mockPeer);

      const config = await peer.getPeerConfig();

      expect(config).toEqual({ observe_me: true, observe_others: false });
      expect(mockClient.workspaces.peers.getOrCreate).toHaveBeenCalledWith(
        'test-workspace',
        { id: 'test-peer' }
      );
    });

    it('should return empty object when no configuration exists', async () => {
      const mockPeer = {
        id: 'test-peer',
        configuration: null,
      };
      mockClient.workspaces.peers.getOrCreate.mockResolvedValue(mockPeer);

      const config = await peer.getPeerConfig();

      expect(config).toEqual({});
    });
  });

  describe('setPeerConfig', () => {
    it('should update peer configuration', async () => {
      const config = { observe_me: false, observe_others: true };
      mockClient.workspaces.peers.update.mockResolvedValue({});

      await peer.setPeerConfig(config);

      expect(mockClient.workspaces.peers.update).toHaveBeenCalledWith(
        'test-workspace',
        'test-peer',
        { configuration: config }
      );
    });
  });

  describe('search', () => {
    it('should search peer messages and return array', async () => {
      const mockSearchResults = [
        { id: 'msg1', content: 'Hello world', peer_id: 'test-peer' },
        { id: 'msg2', content: 'Hello there', peer_id: 'test-peer' },
      ];
      mockClient.workspaces.peers.search.mockResolvedValue(mockSearchResults);

      const results = await peer.search('hello');

      expect(Array.isArray(results)).toBe(true);
      expect(mockClient.workspaces.peers.search).toHaveBeenCalledWith(
        'test-workspace',
        'test-peer',
        { query: 'hello' }
      );
    });

    it('should handle empty search results', async () => {
      const mockSearchResults: any[] = [];
      mockClient.workspaces.peers.search.mockResolvedValue(mockSearchResults);

      const results = await peer.search('nonexistent');

      expect(Array.isArray(results)).toBe(true);
    });

    it('should throw error for empty query', async () => {
      await expect(peer.search('')).rejects.toThrow();
      await expect(peer.search('   ')).rejects.toThrow();
    });

    it('should throw error for non-string query', async () => {
      await expect(peer.search(null as any)).rejects.toThrow();
      await expect(peer.search(undefined as any)).rejects.toThrow();
      await expect(peer.search(123 as any)).rejects.toThrow();
    });

    it('should handle complex search queries', async () => {
      const mockSearchResults: any[] = [];
      mockClient.workspaces.peers.search.mockResolvedValue(mockSearchResults);

      const complexQuery = 'complex query with "quotes" and special characters!@#$%';
      await peer.search(complexQuery);

      expect(mockClient.workspaces.peers.search).toHaveBeenCalledWith(
        'test-workspace',
        'test-peer',
        { query: complexQuery }
      );
    });

    it('should handle API errors', async () => {
      mockClient.workspaces.peers.search.mockRejectedValue(new Error('Search failed'));

      await expect(peer.search('test')).rejects.toThrow();
    });
  });

  describe('card', () => {
    beforeEach(() => {
      mockClient.workspaces.peers.card = jest.fn();
    });

    it('should get peer card without target', async () => {
      const mockCardResponse = {
        peer_card: ['Fact 1 about peer', 'Fact 2 about peer', 'Fact 3 about peer'],
      };
      mockClient.workspaces.peers.card.mockResolvedValue(mockCardResponse);

      const result = await peer.card();

      expect(result).toBe('Fact 1 about peer\nFact 2 about peer\nFact 3 about peer');
      expect(mockClient.workspaces.peers.card).toHaveBeenCalledWith(
        'test-workspace',
        'test-peer',
        { target: undefined }
      );
    });

    it('should get peer card with target as string', async () => {
      const mockCardResponse = {
        peer_card: ['What peer knows about target'],
      };
      mockClient.workspaces.peers.card.mockResolvedValue(mockCardResponse);

      const result = await peer.card('target-peer');

      expect(result).toBe('What peer knows about target');
      expect(mockClient.workspaces.peers.card).toHaveBeenCalledWith(
        'test-workspace',
        'test-peer',
        { target: 'target-peer' }
      );
    });

    it('should get peer card with target as Peer object', async () => {
      const targetPeer = new Peer('target-peer', 'test-workspace', mockClient);
      const mockCardResponse = {
        peer_card: ['What peer knows about target peer'],
      };
      mockClient.workspaces.peers.card.mockResolvedValue(mockCardResponse);

      const result = await peer.card(targetPeer);

      expect(result).toBe('What peer knows about target peer');
      expect(mockClient.workspaces.peers.card).toHaveBeenCalledWith(
        'test-workspace',
        'test-peer',
        { target: 'target-peer' }
      );
    });

    it('should return empty string when peer_card is null', async () => {
      const mockCardResponse = {
        peer_card: null,
      };
      mockClient.workspaces.peers.card.mockResolvedValue(mockCardResponse);

      const result = await peer.card();

      expect(result).toBe('');
    });

    it('should return empty string when peer_card is undefined', async () => {
      const mockCardResponse = {};
      mockClient.workspaces.peers.card.mockResolvedValue(mockCardResponse);

      const result = await peer.card();

      expect(result).toBe('');
    });

    it('should throw error for empty string target', async () => {
      await expect(peer.card('')).rejects.toThrow('target string cannot be empty');
      await expect(peer.card('   ')).rejects.toThrow('target string cannot be empty');
    });

    it('should throw error for invalid target type', async () => {
      await expect(peer.card(123 as any)).rejects.toThrow('target must be string, Peer, or undefined');
      await expect(peer.card(null as any)).rejects.toThrow('target must be string, Peer, or undefined');
      await expect(peer.card({} as any)).rejects.toThrow('target must be string, Peer, or undefined');
    });

    it('should handle API errors', async () => {
      mockClient.workspaces.peers.card.mockRejectedValue(new Error('Card fetch failed'));

      await expect(peer.card()).rejects.toThrow('Card fetch failed');
    });
  });

  describe('workingRep', () => {
    beforeEach(() => {
      mockClient.workspaces.peers.workingRepresentation = jest.fn();
    });

    it('should get working representation with no parameters', async () => {
      const mockRepresentationData = {
        explicit: [
          {
            content: 'Observation 1',
            created_at: '2024-01-01T00:00:00Z',
            message_ids: [[1, 2]],
            session_name: 'test-session',
          },
          {
            content: 'Observation 2',
            created_at: '2024-01-01T00:01:00Z',
            message_ids: [[3, 4]],
            session_name: 'test-session',
          },
        ],
        deductive: [
          {
            conclusion: 'Conclusion 1',
            premises: ['Observation 1', 'Observation 2'],
            created_at: '2024-01-01T00:02:00Z',
            message_ids: [[5, 6]],
            session_name: 'test-session',
          },
        ],
      };
      mockClient.workspaces.peers.workingRepresentation.mockResolvedValue({
        representation: mockRepresentationData,
      });

      const result = await peer.workingRep();

      expect(result).toBeInstanceOf(Representation);
      expect(result.explicit).toHaveLength(2);
      expect(result.explicit[0].content).toBe('Observation 1');
      expect(result.explicit[1].content).toBe('Observation 2');
      expect(result.deductive).toHaveLength(1);
      expect(result.deductive[0].conclusion).toBe('Conclusion 1');
      expect(
        mockClient.workspaces.peers.workingRepresentation
      ).toHaveBeenCalledWith('test-workspace', 'test-peer', {
        session_id: undefined,
        target: undefined,
        search_query: undefined,
        search_top_k: undefined,
        search_max_distance: undefined,
        include_most_derived: undefined,
        max_observations: undefined,
      });
    });

    it('should get working representation with session as string', async () => {
      const mockRepresentationData = {
        explicit: [
          {
            content: 'Session-scoped observation',
            created_at: '2024-01-01T00:00:00Z',
            message_ids: [[1, 2]],
            session_name: 'session-123',
          },
        ],
        deductive: [],
      };
      mockClient.workspaces.peers.workingRepresentation.mockResolvedValue({
        representation: mockRepresentationData,
      });

      const result = await peer.workingRep('session-123');

      expect(result).toBeInstanceOf(Representation);
      expect(result.explicit).toHaveLength(1);
      expect(result.explicit[0].content).toBe('Session-scoped observation');
      expect(result.deductive).toHaveLength(0);
      expect(
        mockClient.workspaces.peers.workingRepresentation
      ).toHaveBeenCalledWith('test-workspace', 'test-peer', {
        session_id: 'session-123',
        target: undefined,
        search_query: undefined,
        search_top_k: undefined,
        search_max_distance: undefined,
        include_most_derived: undefined,
        max_observations: undefined,
      });
    });

    it('should get working representation with session as Session object', async () => {
      const session = new Session('session-123', 'test-workspace', mockClient);
      const mockRepresentationData = {
        explicit: [
          {
            content: 'Session object observation',
            created_at: '2024-01-01T00:00:00Z',
            message_ids: [[1, 2]],
            session_name: 'session-123',
          },
        ],
        deductive: [],
      };
      mockClient.workspaces.peers.workingRepresentation.mockResolvedValue({
        representation: mockRepresentationData,
      });

      const result = await peer.workingRep(session);

      expect(result).toBeInstanceOf(Representation);
      expect(result.explicit).toHaveLength(1);
      expect(result.explicit[0].content).toBe('Session object observation');
      expect(result.deductive).toHaveLength(0);
      expect(
        mockClient.workspaces.peers.workingRepresentation
      ).toHaveBeenCalledWith('test-workspace', 'test-peer', {
        session_id: 'session-123',
        target: undefined,
        search_query: undefined,
        search_top_k: undefined,
        search_max_distance: undefined,
        include_most_derived: undefined,
        max_observations: undefined,
      });
    });

    it('should get working representation with target as string', async () => {
      const mockRepresentationData = {
        explicit: [
          {
            content: "Observer's view of target",
            created_at: '2024-01-01T00:00:00Z',
            message_ids: [[1, 2]],
            session_name: 'test-session',
          },
        ],
        deductive: [],
      };
      mockClient.workspaces.peers.workingRepresentation.mockResolvedValue({
        representation: mockRepresentationData,
      });

      const result = await peer.workingRep(undefined, 'target-peer');

      expect(result).toBeInstanceOf(Representation);
      expect(result.explicit).toHaveLength(1);
      expect(result.explicit[0].content).toBe("Observer's view of target");
      expect(result.deductive).toHaveLength(0);
      expect(
        mockClient.workspaces.peers.workingRepresentation
      ).toHaveBeenCalledWith('test-workspace', 'test-peer', {
        session_id: undefined,
        target: 'target-peer',
        search_query: undefined,
        search_top_k: undefined,
        search_max_distance: undefined,
        include_most_derived: undefined,
        max_observations: undefined,
      });
    });

    it('should get working representation with target as Peer object', async () => {
      const targetPeer = new Peer('target-peer', 'test-workspace', mockClient);
      const mockRepresentationData = {
        explicit: [
          {
            content: "Observer's view of target peer object",
            created_at: '2024-01-01T00:00:00Z',
            message_ids: [[1, 2]],
            session_name: 'test-session',
          },
        ],
        deductive: [],
      };
      mockClient.workspaces.peers.workingRepresentation.mockResolvedValue({
        representation: mockRepresentationData,
      });

      const result = await peer.workingRep(undefined, targetPeer);

      expect(result).toBeInstanceOf(Representation);
      expect(result.explicit).toHaveLength(1);
      expect(result.explicit[0].content).toBe("Observer's view of target peer object");
      expect(result.deductive).toHaveLength(0);
      expect(
        mockClient.workspaces.peers.workingRepresentation
      ).toHaveBeenCalledWith('test-workspace', 'test-peer', {
        session_id: undefined,
        target: 'target-peer',
        search_query: undefined,
        search_top_k: undefined,
        search_max_distance: undefined,
        include_most_derived: undefined,
        max_observations: undefined,
      });
    });

    it('should get working representation with search query', async () => {
      const mockRepresentationData = {
        explicit: [
          {
            content: 'Query-curated observation',
            created_at: '2024-01-01T00:00:00Z',
            message_ids: [[1, 2]],
            session_name: 'test-session',
          },
        ],
        deductive: [],
      };
      mockClient.workspaces.peers.workingRepresentation.mockResolvedValue({
        representation: mockRepresentationData,
      });

      const result = await peer.workingRep(
        undefined,
        undefined,
        { searchQuery: 'programming' }
      );

      expect(result).toBeInstanceOf(Representation);
      expect(result.explicit).toHaveLength(1);
      expect(result.explicit[0].content).toBe('Query-curated observation');
      expect(result.deductive).toHaveLength(0);
      expect(
        mockClient.workspaces.peers.workingRepresentation
      ).toHaveBeenCalledWith('test-workspace', 'test-peer', {
        session_id: undefined,
        target: undefined,
        search_query: 'programming',
        search_top_k: undefined,
        search_max_distance: undefined,
        include_most_derived: undefined,
        max_observations: undefined,
      });
    });

    it('should get working representation with custom size', async () => {
      const mockRepresentationData = {
        explicit: [
          {
            content: 'Limited observations',
            created_at: '2024-01-01T00:00:00Z',
            message_ids: [[1, 2]],
            session_name: 'test-session',
          },
        ],
        deductive: [],
      };
      mockClient.workspaces.peers.workingRepresentation.mockResolvedValue({
        representation: mockRepresentationData,
      });

      const result = await peer.workingRep(undefined, undefined, { maxObservations: 10 });

      expect(result).toBeInstanceOf(Representation);
      expect(result.explicit).toHaveLength(1);
      expect(result.explicit[0].content).toBe('Limited observations');
      expect(result.deductive).toHaveLength(0);
      expect(
        mockClient.workspaces.peers.workingRepresentation
      ).toHaveBeenCalledWith('test-workspace', 'test-peer', {
        session_id: undefined,
        target: undefined,
        search_query: undefined,
        search_top_k: undefined,
        search_max_distance: undefined,
        include_most_derived: undefined,
        max_observations: 10,
      });
    });

    it('should get working representation with all parameters', async () => {
      const session = new Session('session-123', 'test-workspace', mockClient);
      const targetPeer = new Peer('target-peer', 'test-workspace', mockClient);
      const mockRepresentationData = {
        explicit: [
          {
            content: 'Fully parameterized observation',
            created_at: '2024-01-01T00:00:00Z',
            message_ids: [[1, 2]],
            session_name: 'session-123',
          },
        ],
        deductive: [
          {
            conclusion: 'Conclusion with all params',
            premises: ['Fully parameterized observation'],
            created_at: '2024-01-01T00:01:00Z',
            message_ids: [[3, 4]],
            session_name: 'session-123',
          },
        ],
      };
      mockClient.workspaces.peers.workingRepresentation.mockResolvedValue({
        representation: mockRepresentationData,
      });

      const result = await peer.workingRep(
        session,
        targetPeer,
        { searchQuery: 'Python programming', maxObservations: 25 }
      );

      expect(result).toBeInstanceOf(Representation);
      expect(result.explicit).toHaveLength(1);
      expect(result.explicit[0].content).toBe('Fully parameterized observation');
      expect(result.deductive).toHaveLength(1);
      expect(result.deductive[0].conclusion).toBe('Conclusion with all params');
      expect(
        mockClient.workspaces.peers.workingRepresentation
      ).toHaveBeenCalledWith('test-workspace', 'test-peer', {
        session_id: 'session-123',
        target: 'target-peer',
        search_query: 'Python programming',
        search_top_k: undefined,
        search_max_distance: undefined,
        include_most_derived: undefined,
        max_observations: 25,
      });
    });

    it('should get working representation with string session and string target', async () => {
      const mockRepresentationData = {
        explicit: [
          {
            content: 'String params observation',
            created_at: '2024-01-01T00:00:00Z',
            message_ids: [[1, 2]],
            session_name: 'session-456',
          },
        ],
        deductive: [],
      };
      mockClient.workspaces.peers.workingRepresentation.mockResolvedValue({
        representation: mockRepresentationData,
      });

      const result = await peer.workingRep(
        'session-456',
        'target-peer-123',
        { searchQuery: 'machine learning', maxObservations: 50 }
      );

      expect(result).toBeInstanceOf(Representation);
      expect(result.explicit).toHaveLength(1);
      expect(result.explicit[0].content).toBe('String params observation');
      expect(result.deductive).toHaveLength(0);
      expect(
        mockClient.workspaces.peers.workingRepresentation
      ).toHaveBeenCalledWith('test-workspace', 'test-peer', {
        session_id: 'session-456',
        target: 'target-peer-123',
        search_query: 'machine learning',
        search_top_k: undefined,
        search_max_distance: undefined,
        include_most_derived: undefined,
        max_observations: 50,
      });
    });

    it('should handle boundary size values', async () => {
      const mockRepresentationData = {
        explicit: [
          {
            content: 'Boundary test',
            created_at: '2024-01-01T00:00:00Z',
            message_ids: [[1, 2]],
            session_name: 'test-session',
          },
        ],
        deductive: [],
      };
      mockClient.workspaces.peers.workingRepresentation.mockResolvedValue({
        representation: mockRepresentationData,
      });

      // Test size = 1
      const result1 = await peer.workingRep(undefined, undefined, { maxObservations: 1 });
      expect(result1).toBeInstanceOf(Representation);
      expect(
        mockClient.workspaces.peers.workingRepresentation
      ).toHaveBeenLastCalledWith('test-workspace', 'test-peer', {
        session_id: undefined,
        target: undefined,
        search_query: undefined,
        search_top_k: undefined,
        search_max_distance: undefined,
        include_most_derived: undefined,
        max_observations: 1,
      });

      // Test size = 100
      const result2 = await peer.workingRep(undefined, undefined, { maxObservations: 100 });
      expect(result2).toBeInstanceOf(Representation);
      expect(
        mockClient.workspaces.peers.workingRepresentation
      ).toHaveBeenLastCalledWith('test-workspace', 'test-peer', {
        session_id: undefined,
        target: undefined,
        search_query: undefined,
        search_top_k: undefined,
        search_max_distance: undefined,
        include_most_derived: undefined,
        max_observations: 100,
      });
    });

    it('should handle API errors', async () => {
      mockClient.workspaces.peers.workingRepresentation.mockRejectedValue(
        new Error('Working representation fetch failed')
      );

      await expect(peer.workingRep()).rejects.toThrow(
        'Working representation fetch failed'
      );
    });
  });
});
