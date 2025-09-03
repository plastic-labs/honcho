import { beforeEach, describe, expect, it, jest } from '@jest/globals';
import { Honcho } from '../src/client';
import { Page } from '../src/pagination';
import { Peer } from '../src/peer';
import { Session } from '../src/session';
import { SessionContext } from '../src/session_context';

// Mock the @honcho-ai/core module
jest.mock('@honcho-ai/core', () => {
  return jest.fn().mockImplementation(() => ({
    workspaces: {
      sessions: {
        peers: {
          add: jest.fn(),
          set: jest.fn(),
          remove: jest.fn(),
          list: jest.fn(),
          getConfig: jest.fn(),
          setConfig: jest.fn(),
        },
        messages: {
          create: jest.fn(),
          list: jest.fn(),
          upload: jest.fn(),
          update: jest.fn(),
        },
        getOrCreate: jest.fn(),
        update: jest.fn(),
        delete: jest.fn(),
        getContext: jest.fn(),
        search: jest.fn(),
      },
      peers: {
        workingRepresentation: jest.fn(),
      },
      deriverStatus: jest.fn(),
      getOrCreate: jest.fn(),
      update: jest.fn(),
      list: jest.fn(),
      search: jest.fn(),
    },
  }));
});

describe('Session', () => {
  let honcho: Honcho;
  let session: Session;
  let mockClient: any;

  beforeEach(() => {
    jest.clearAllMocks();

    honcho = new Honcho({
      workspaceId: 'test-workspace',
      apiKey: 'test-key',
      environment: 'local',
    });

    session = new Session('test-session', 'test-workspace', (honcho as any)._client);
    mockClient = (honcho as any)._client;
  });

  describe('constructor', () => {
    it('should initialize with correct properties', () => {
      const newSession = new Session('session-id', 'test-workspace', mockClient);

      expect(newSession.id).toBe('session-id');
      expect(newSession.workspaceId).toBe('test-workspace');
      expect(newSession['_client']).toBe(mockClient);
    });
  });

  describe('addPeers', () => {
    it('should add single peer by string ID', async () => {
      mockClient.workspaces.sessions.peers.add.mockResolvedValue({});

      await session.addPeers('peer1');

      expect(mockClient.workspaces.sessions.peers.add).toHaveBeenCalledWith(
        'test-workspace',
        'test-session',
        { 'peer1': {} }
      );
    });

    it('should add single peer by Peer object', async () => {
      const peer = new Peer('peer1', 'test-workspace', mockClient);
      mockClient.workspaces.sessions.peers.add.mockResolvedValue({});

      await session.addPeers(peer);

      expect(mockClient.workspaces.sessions.peers.add).toHaveBeenCalledWith(
        'test-workspace',
        'test-session',
        { 'peer1': {} }
      );
    });

    it('should add array of peer strings', async () => {
      mockClient.workspaces.sessions.peers.add.mockResolvedValue({});

      await session.addPeers(['peer1', 'peer2', 'peer3']);

      expect(mockClient.workspaces.sessions.peers.add).toHaveBeenCalledWith(
        'test-workspace',
        'test-session',
        {
          'peer1': {},
          'peer2': {},
          'peer3': {}
        }
      );
    });

    it('should add array of Peer objects', async () => {
      const peers = [
        new Peer('peer1', 'test-workspace', mockClient),
        new Peer('peer2', 'test-workspace', mockClient),
        new Peer('peer3', 'test-workspace', mockClient),
      ];
      mockClient.workspaces.sessions.peers.add.mockResolvedValue({});

      await session.addPeers(peers);

      expect(mockClient.workspaces.sessions.peers.add).toHaveBeenCalledWith(
        'test-workspace',
        'test-session',
        {
          'peer1': {},
          'peer2': {},
          'peer3': {}
        }
      );
    });

    it('should add mixed array of strings and Peer objects', async () => {
      const peers = [
        'string-peer',
        new Peer('object-peer', 'test-workspace', mockClient),
      ];
      mockClient.workspaces.sessions.peers.add.mockResolvedValue({});

      await session.addPeers(peers);

      expect(mockClient.workspaces.sessions.peers.add).toHaveBeenCalledWith(
        'test-workspace',
        'test-session',
        {
          'string-peer': {},
          'object-peer': {}
        }
      );
    });

    it('should add peer with SessionPeerConfig', async () => {
      const { SessionPeerConfig } = require('../src/session');
      const config = new SessionPeerConfig(false, true);
      mockClient.workspaces.sessions.peers.add.mockResolvedValue({});

      await session.addPeers([['peer1', config]]);

      expect(mockClient.workspaces.sessions.peers.add).toHaveBeenCalledWith(
        'test-workspace',
        'test-session',
        {
          'peer1': { observe_me: false, observe_others: true }
        }
      );
    });

    it('should handle API errors', async () => {
      mockClient.workspaces.sessions.peers.add.mockRejectedValue(new Error('Failed to add peers'));

      await expect(session.addPeers('peer1')).rejects.toThrow();
    });
  });

  describe('setPeers', () => {
    it('should set single peer by string ID', async () => {
      mockClient.workspaces.sessions.peers.set.mockResolvedValue({});

      await session.setPeers('peer1');

      expect(mockClient.workspaces.sessions.peers.set).toHaveBeenCalledWith(
        'test-workspace',
        'test-session',
        { 'peer1': {} }
      );
    });

    it('should set single peer by Peer object', async () => {
      const peer = new Peer('peer1', 'test-workspace', mockClient);
      mockClient.workspaces.sessions.peers.set.mockResolvedValue({});

      await session.setPeers(peer);

      expect(mockClient.workspaces.sessions.peers.set).toHaveBeenCalledWith(
        'test-workspace',
        'test-session',
        { 'peer1': {} }
      );
    });

    it('should set array of peers', async () => {
      const peers = ['peer1', new Peer('peer2', 'test-workspace', mockClient)];
      mockClient.workspaces.sessions.peers.set.mockResolvedValue({});

      await session.setPeers(peers);

      expect(mockClient.workspaces.sessions.peers.set).toHaveBeenCalledWith(
        'test-workspace',
        'test-session',
        {
          'peer1': {},
          'peer2': {}
        }
      );
    });

    it('should handle API errors', async () => {
      mockClient.workspaces.sessions.peers.set.mockRejectedValue(new Error('Failed to set peers'));

      await expect(session.setPeers(['peer1'])).rejects.toThrow();
    });
  });

  describe('removePeers', () => {
    it('should remove single peer by string ID', async () => {
      mockClient.workspaces.sessions.peers.remove.mockResolvedValue({});

      await session.removePeers('peer1');

      expect(mockClient.workspaces.sessions.peers.remove).toHaveBeenCalledWith(
        'test-workspace',
        'test-session',
        ['peer1']
      );
    });

    it('should remove single peer by Peer object', async () => {
      const peer = new Peer('peer1', 'test-workspace', mockClient);
      mockClient.workspaces.sessions.peers.remove.mockResolvedValue({});

      await session.removePeers(peer);

      expect(mockClient.workspaces.sessions.peers.remove).toHaveBeenCalledWith(
        'test-workspace',
        'test-session',
        ['peer1']
      );
    });

    it('should remove array of peers', async () => {
      const peers = ['peer1', new Peer('peer2', 'test-workspace', mockClient)];
      mockClient.workspaces.sessions.peers.remove.mockResolvedValue({});

      await session.removePeers(peers);

      expect(mockClient.workspaces.sessions.peers.remove).toHaveBeenCalledWith(
        'test-workspace',
        'test-session',
        ['peer1', 'peer2']
      );
    });

    it('should handle API errors', async () => {
      mockClient.workspaces.sessions.peers.remove.mockRejectedValue(new Error('Failed to remove peers'));

      await expect(session.removePeers(['peer1'])).rejects.toThrow();
    });
  });

  describe('getPeers', () => {
    it('should return array of Peer instances', async () => {
      const mockPeersData = {
        items: [
          { id: 'peer1', metadata: {} },
          { id: 'peer2', metadata: {} },
        ],
        total: 2,
        size: 2,
        hasNextPage: false,
      };
      mockClient.workspaces.sessions.peers.list.mockResolvedValue(mockPeersData);

      const peers = await session.getPeers();

      expect(peers).toBeInstanceOf(Array);
      expect(peers).toHaveLength(2);
      expect(peers[0]).toBeInstanceOf(Peer);
      expect(peers[1]).toBeInstanceOf(Peer);
      expect(mockClient.workspaces.sessions.peers.list).toHaveBeenCalledWith(
        'test-workspace',
        'test-session'
      );
    });

    it('should handle empty peers list', async () => {
      const mockPeersData = {
        items: [],
        total: 0,
        size: 0,
        hasNextPage: false,
      };
      mockClient.workspaces.sessions.peers.list.mockResolvedValue(mockPeersData);

      const peers = await session.getPeers();

      expect(peers).toBeInstanceOf(Array);
      expect(peers.length).toBe(0);
    });

    it('should handle API errors', async () => {
      mockClient.workspaces.sessions.peers.list.mockRejectedValue(new Error('Failed to get peers'));

      await expect(session.getPeers()).rejects.toThrow();
    });
  });

  describe('getPeerConfig', () => {
    it('should return peer configuration', async () => {
      const mockConfig = { observe_me: true, observe_others: false };
      mockClient.workspaces.sessions.peers.getConfig.mockResolvedValue(mockConfig);

      const config = await session.getPeerConfig('peer1');

      expect(config).toEqual(mockConfig);
      expect(mockClient.workspaces.sessions.peers.getConfig).toHaveBeenCalledWith(
        'test-workspace',
        'test-session',
        'peer1'
      );
    });

    it('should handle Peer object input', async () => {
      const peer = new Peer('peer1', 'test-workspace', mockClient);
      const mockConfig = { observe_me: false, observe_others: true };
      mockClient.workspaces.sessions.peers.getConfig.mockResolvedValue(mockConfig);

      const config = await session.getPeerConfig(peer);

      expect(config).toEqual(mockConfig);
      expect(mockClient.workspaces.sessions.peers.getConfig).toHaveBeenCalledWith(
        'test-workspace',
        'test-session',
        'peer1'
      );
    });
  });

  describe('setPeerConfig', () => {
    it('should set peer configuration', async () => {
      const { SessionPeerConfig } = require('../src/session');
      const config = new SessionPeerConfig(false, true);
      mockClient.workspaces.sessions.peers.setConfig.mockResolvedValue({});

      await session.setPeerConfig('peer1', config);

      expect(mockClient.workspaces.sessions.peers.setConfig).toHaveBeenCalledWith(
        'test-workspace',
        'test-session',
        'peer1',
        { observe_me: false, observe_others: true }
      );
    });

    it('should handle Peer object input', async () => {
      const peer = new Peer('peer1', 'test-workspace', mockClient);
      const { SessionPeerConfig } = require('../src/session');
      const config = new SessionPeerConfig(true, false);
      mockClient.workspaces.sessions.peers.setConfig.mockResolvedValue({});

      await session.setPeerConfig(peer, config);

      expect(mockClient.workspaces.sessions.peers.setConfig).toHaveBeenCalledWith(
        'test-workspace',
        'test-session',
        'peer1',
        { observe_me: true, observe_others: false }
      );
    });
  });

  describe('addMessages', () => {
    it('should add single message', async () => {
      const message = {
        peer_id: 'peer1',
        content: 'Hello world',
        metadata: { type: 'greeting' },
      };
      mockClient.workspaces.sessions.messages.create.mockResolvedValue({});

      await session.addMessages(message);

      expect(mockClient.workspaces.sessions.messages.create).toHaveBeenCalledWith(
        'test-workspace',
        'test-session',
        {
          messages: [{
            peer_id: 'peer1',
            content: 'Hello world',
            metadata: { type: 'greeting' }
          }]
        }
      );
    });

    it('should add array of messages', async () => {
      const messages = [
        { peer_id: 'peer1', content: 'Message 1', metadata: { order: 1 } },
        { peer_id: 'peer2', content: 'Message 2', metadata: { order: 2 } },
      ];
      mockClient.workspaces.sessions.messages.create.mockResolvedValue({});

      await session.addMessages(messages);

      expect(mockClient.workspaces.sessions.messages.create).toHaveBeenCalledWith(
        'test-workspace',
        'test-session',
        {
          messages: [
            { peer_id: 'peer1', content: 'Message 1', metadata: { order: 1 } },
            { peer_id: 'peer2', content: 'Message 2', metadata: { order: 2 } },
          ]
        }
      );
    });

    it('should handle messages without metadata', async () => {
      const message = {
        peer_id: 'peer1',
        content: 'Simple message',
      };
      mockClient.workspaces.sessions.messages.create.mockResolvedValue({});

      await session.addMessages(message);

      expect(mockClient.workspaces.sessions.messages.create).toHaveBeenCalledWith(
        'test-workspace',
        'test-session',
        { messages: [{ peer_id: 'peer1', content: 'Simple message' }] }
      );
    });

    it('should handle empty array', async () => {
      mockClient.workspaces.sessions.messages.create.mockResolvedValue({});

      await session.addMessages([]);

      expect(mockClient.workspaces.sessions.messages.create).toHaveBeenCalledWith(
        'test-workspace',
        'test-session',
        { messages: [] }
      );
    });

    it('should handle API errors', async () => {
      mockClient.workspaces.sessions.messages.create.mockRejectedValue(new Error('Failed to add messages'));

      await expect(session.addMessages({ peer_id: 'peer1', content: 'test' })).rejects.toThrow();
    });

    it('should add message with custom timestamp', async () => {
      const message = {
        peer_id: 'peer1',
        content: 'Message with timestamp',
        created_at: '2023-01-01T12:00:00Z',
        metadata: { test: 'timestamp' },
      };
      mockClient.workspaces.sessions.messages.create.mockResolvedValue({});

      await session.addMessages(message);

      expect(mockClient.workspaces.sessions.messages.create).toHaveBeenCalledWith(
        'test-workspace',
        'test-session',
        {
          messages: [{
            peer_id: 'peer1',
            content: 'Message with timestamp',
            created_at: '2023-01-01T12:00:00Z',
            metadata: { test: 'timestamp' }
          }]
        }
      );
    });

    it('should add message with null timestamp', async () => {
      const message = {
        peer_id: 'peer1',
        content: 'Message without timestamp',
        created_at: null,
        metadata: { test: 'no_timestamp' },
      };
      mockClient.workspaces.sessions.messages.create.mockResolvedValue({});

      await session.addMessages(message);

      expect(mockClient.workspaces.sessions.messages.create).toHaveBeenCalledWith(
        'test-workspace',
        'test-session',
        {
          messages: [{
            peer_id: 'peer1',
            content: 'Message without timestamp',
            created_at: null,
            metadata: { test: 'no_timestamp' }
          }]
        }
      );
    });

    it('should add mixed messages with and without timestamps', async () => {
      const messages = [
        {
          peer_id: 'peer1',
          content: 'Message with timestamp',
          created_at: '2023-01-01T12:00:00Z',
          metadata: { type: 'historical' }
        },
        {
          peer_id: 'peer2',
          content: 'Message without timestamp',
          metadata: { type: 'current' }
        },
        {
          peer_id: 'peer3',
          content: 'Message with null timestamp',
          created_at: null,
          metadata: { type: 'default' }
        }
      ];
      mockClient.workspaces.sessions.messages.create.mockResolvedValue({});

      await session.addMessages(messages);

      expect(mockClient.workspaces.sessions.messages.create).toHaveBeenCalledWith(
        'test-workspace',
        'test-session',
        {
          messages: [
            {
              peer_id: 'peer1',
              content: 'Message with timestamp',
              created_at: '2023-01-01T12:00:00Z',
              metadata: { type: 'historical' }
            },
            {
              peer_id: 'peer2',
              content: 'Message without timestamp',
              metadata: { type: 'current' }
            },
            {
              peer_id: 'peer3',
              content: 'Message with null timestamp',
              created_at: null,
              metadata: { type: 'default' }
            }
          ]
        }
      );
    });
  });

  describe('getMessages', () => {
    it('should get messages without filter', async () => {
      const mockMessagesData = {
        items: [
          { id: 'msg1', content: 'Message 1', peer_id: 'peer1' },
          { id: 'msg2', content: 'Message 2', peer_id: 'peer2' },
        ],
        total: 2,
        size: 2,
        hasNextPage: false,
      };
      mockClient.workspaces.sessions.messages.list.mockResolvedValue(mockMessagesData);

      const messagesPage = await session.getMessages();

      expect(messagesPage).toBeInstanceOf(Page);
      expect(mockClient.workspaces.sessions.messages.list).toHaveBeenCalledWith(
        'test-workspace',
        'test-session',
        undefined
      );
    });

    it('should get messages with filter options', async () => {
      const mockMessagesData = {
        items: [],
        total: 0,
        size: 0,
        hasNextPage: false,
      };
      mockClient.workspaces.sessions.messages.list.mockResolvedValue(mockMessagesData);

      const filter = { peer_id: { value: 'peer1' }, type: { value: 'important' } };
      await session.getMessages(filter);

      expect(mockClient.workspaces.sessions.messages.list).toHaveBeenCalledWith(
        'test-workspace',
        'test-session',
        { peer_id: { value: 'peer1' }, type: { value: 'important' } }
      );
    });

    it('should handle API errors', async () => {
      mockClient.workspaces.sessions.messages.list.mockRejectedValue(new Error('Failed to get messages'));

      await expect(session.getMessages()).rejects.toThrow();
    });
  });

  describe('getMetadata', () => {
    it('should return session metadata', async () => {
      const mockSession = {
        id: 'test-session',
        metadata: { name: 'Test Session', active: true },
      };
      mockClient.workspaces.sessions.getOrCreate.mockResolvedValue(mockSession);

      const metadata = await session.getMetadata();

      expect(metadata).toEqual({ name: 'Test Session', active: true });
      expect(mockClient.workspaces.sessions.getOrCreate).toHaveBeenCalledWith(
        'test-workspace',
        { id: 'test-session' }
      );
    });

    it('should return empty object when no metadata exists', async () => {
      const mockSession = {
        id: 'test-session',
        metadata: null,
      };
      mockClient.workspaces.sessions.getOrCreate.mockResolvedValue(mockSession);

      const metadata = await session.getMetadata();

      expect(metadata).toEqual({});
    });

    it('should handle API errors', async () => {
      mockClient.workspaces.sessions.getOrCreate.mockRejectedValue(new Error('Session not found'));

      await expect(session.getMetadata()).rejects.toThrow();
    });
  });

  describe('setMetadata', () => {
    it('should update session metadata', async () => {
      const metadata = { name: 'Updated Session', status: 'active' };
      mockClient.workspaces.sessions.update.mockResolvedValue({});

      await session.setMetadata(metadata);

      expect(mockClient.workspaces.sessions.update).toHaveBeenCalledWith(
        'test-workspace',
        'test-session',
        { metadata }
      );
    });

    it('should handle empty metadata', async () => {
      mockClient.workspaces.sessions.update.mockResolvedValue({});

      await session.setMetadata({});

      expect(mockClient.workspaces.sessions.update).toHaveBeenCalledWith(
        'test-workspace',
        'test-session',
        { metadata: {} }
      );
    });

    it('should handle API errors', async () => {
      mockClient.workspaces.sessions.update.mockRejectedValue(new Error('Update failed'));

      await expect(session.setMetadata({ key: 'value' })).rejects.toThrow();
    });
  });

  describe('getContext', () => {
    it('should get session context without options', async () => {
      const mockContext = {
        messages: [
          { id: 'msg1', content: 'Hello', peer_id: 'peer1' },
          { id: 'msg2', content: 'Hi there', peer_id: 'peer2' },
        ],
        summary: {
          content: 'Conversation summary',
          message_id: '10',
          summary_type: 'short',
          created_at: '2024-01-01T00:00:00Z',
          token_count: 100
        },
      };
      mockClient.workspaces.sessions.getContext.mockResolvedValue(mockContext);

      const context = await session.getContext();

      expect(context).toBeInstanceOf(SessionContext);
      expect(context.sessionId).toBe('test-session');
      expect(context.messages).toHaveLength(2);
      expect(context.messages[0].id).toBe('msg1');
      expect(context.messages[0].content).toBe('Hello');
      expect(context.messages[0].peer_id).toBe('peer1');
      expect(context.messages[1].id).toBe('msg2');
      expect(context.messages[1].content).toBe('Hi there');
      expect(context.messages[1].peer_id).toBe('peer2');
      expect(context.summary?.content).toBe('Conversation summary');
      expect(mockClient.workspaces.sessions.getContext).toHaveBeenCalledWith(
        'test-workspace',
        'test-session',
        { tokens: undefined, summary: undefined }
      );
    });

    it('should get session context with options', async () => {
      const mockContext = {
        messages: [{ id: 'msg1', content: 'Hello', peer_id: 'peer1' }],
        summary: {
          content: 'Brief summary',
          message_id: 5,
          summary_type: 'short',
          created_at: '2024-01-01T00:00:00Z',
          token_count: 50
        },
      };
      mockClient.workspaces.sessions.getContext.mockResolvedValue(mockContext);

      const context = await session.getContext({ summary: true, tokens: 1000 });

      expect(context).toBeInstanceOf(SessionContext);
      expect(mockClient.workspaces.sessions.getContext).toHaveBeenCalledWith(
        'test-workspace',
        'test-session',
        { tokens: 1000, summary: true }
      );
    });

    it('should handle context without summary', async () => {
      const mockContext = {
        messages: [{ id: 'msg1', content: 'Hello', peer_id: 'peer1' }],
      };
      mockClient.workspaces.sessions.getContext.mockResolvedValue(mockContext);

      const context = await session.getContext();

      expect(context.summary).toBeNull();
    });

    it('should handle API errors', async () => {
      mockClient.workspaces.sessions.getContext.mockRejectedValue(new Error('Failed to get context'));

      await expect(session.getContext()).rejects.toThrow();
    });
  });

  describe('search', () => {
    it('should search session messages and return Page', async () => {
      const mockSearchResults = [
        { id: 'msg1', content: 'Hello world', peer_id: 'peer1' },
        { id: 'msg2', content: 'Hello there', peer_id: 'peer2' },
      ];
      mockClient.workspaces.sessions.search.mockResolvedValue(mockSearchResults);

      const results = await session.search('hello');

      expect(Array.isArray(results)).toBe(true);
      expect(mockClient.workspaces.sessions.search).toHaveBeenCalledWith(
        'test-workspace',
        'test-session',
        { query: 'hello', limit: undefined }
      );
    });

    it('should handle empty search results', async () => {
      const mockSearchResults: any[] = [];
      mockClient.workspaces.sessions.search.mockResolvedValue(mockSearchResults);

      const results = await session.search('nonexistent');

      expect(Array.isArray(results)).toBe(true);
    });

    it('should throw error for empty query', async () => {
      await expect(session.search('')).rejects.toThrow();
      await expect(session.search('   ')).rejects.toThrow();
    });

    it('should throw error for non-string query', async () => {
      await expect(session.search(null as any)).rejects.toThrow();
      await expect(session.search(undefined as any)).rejects.toThrow();
      await expect(session.search(123 as any)).rejects.toThrow();
    });

    it('should handle API errors', async () => {
      mockClient.workspaces.sessions.search.mockRejectedValue(new Error('Search failed'));

      await expect(session.search('test')).rejects.toThrow();
    });
  });

  describe('uploadFile', () => {
    it('should upload file and return messages', async () => {
      const mockFile = new File(['test content'], 'test.txt', { type: 'text/plain' });
      const mockMessages = [
        { id: 'msg1', content: 'test content', peer_id: 'peer1' }
      ];
      mockClient.workspaces.sessions.messages.upload.mockResolvedValue(mockMessages);

      const messages = await session.uploadFile(mockFile, 'peer1');

      expect(messages).toHaveLength(1);
      expect(messages[0].id).toBe('msg1');
      expect(messages[0].content).toBe('test content');
      expect(messages[0].peer_id).toBe('peer1');
      expect(mockClient.workspaces.sessions.messages.upload).toHaveBeenCalledWith(
        'test-workspace',
        'test-session',
        { file: mockFile, peer_id: 'peer1' }
      );
    });
  });

  describe('workingRep', () => {
    it('should get working representation with peer string', async () => {
      const mockRepresentation = {
        peer_id: 'peer1',
        knowledge: 'Some knowledge about the peer',
        relationships: ['peer2', 'peer3'],
      };
      mockClient.workspaces.peers.workingRepresentation.mockResolvedValue(mockRepresentation);

      const result = await session.workingRep('peer1');

      expect(result).toEqual(mockRepresentation);
      expect(mockClient.workspaces.peers.workingRepresentation).toHaveBeenCalledWith(
        'test-workspace',
        'peer1',
        { session_id: 'test-session', target: undefined }
      );
    });

    it('should get working representation with Peer object', async () => {
      const peer = new Peer('peer1', 'test-workspace', mockClient);
      const mockRepresentation = {
        peer_id: 'peer1',
        knowledge: 'Some knowledge',
      };
      mockClient.workspaces.peers.workingRepresentation.mockResolvedValue(mockRepresentation);

      const result = await session.workingRep(peer);

      expect(result).toEqual(mockRepresentation);
      expect(mockClient.workspaces.peers.workingRepresentation).toHaveBeenCalledWith(
        'test-workspace',
        'peer1',
        { session_id: 'test-session', target: undefined }
      );
    });

    it('should get working representation with target peer string', async () => {
      const mockRepresentation = {
        peer_id: 'peer1',
        target_knowledge: 'What peer1 knows about target',
      };
      mockClient.workspaces.peers.workingRepresentation.mockResolvedValue(mockRepresentation);

      const result = await session.workingRep('peer1', 'target-peer');

      expect(result).toEqual(mockRepresentation);
      expect(mockClient.workspaces.peers.workingRepresentation).toHaveBeenCalledWith(
        'test-workspace',
        'peer1',
        { session_id: 'test-session', target: 'target-peer' }
      );
    });

    it('should get working representation with target Peer object', async () => {
      const peer = new Peer('peer1', 'test-workspace', mockClient);
      const target = new Peer('target-peer', 'test-workspace', mockClient);
      const mockRepresentation = {
        peer_id: 'peer1',
        target_knowledge: 'What peer1 knows about target',
      };
      mockClient.workspaces.peers.workingRepresentation.mockResolvedValue(mockRepresentation);

      const result = await session.workingRep(peer, target);

      expect(result).toEqual(mockRepresentation);
      expect(mockClient.workspaces.peers.workingRepresentation).toHaveBeenCalledWith(
        'test-workspace',
        'peer1',
        { session_id: 'test-session', target: 'target-peer' }
      );
    });

    it('should handle API errors', async () => {
      mockClient.workspaces.peers.workingRepresentation.mockRejectedValue(new Error('Failed to get working representation'));

      await expect(session.workingRep('peer1')).rejects.toThrow();
    });
  });

  describe('delete', () => {
    it('should delete the session', async () => {
      mockClient.workspaces.sessions.delete.mockResolvedValue({});

      await session.delete();

      expect(mockClient.workspaces.sessions.delete).toHaveBeenCalledWith(
        'test-workspace',
        'test-session'
      );
    });

    it('should handle API errors', async () => {
      mockClient.workspaces.sessions.delete.mockRejectedValue(new Error('Failed to delete session'));

      await expect(session.delete()).rejects.toThrow('Failed to delete session');
    });
  });

  describe('getDeriverStatus', () => {
    it('should return deriver status without options', async () => {
      const mockStatus = {
        total_work_units: 10,
        completed_work_units: 5,
        in_progress_work_units: 3,
        pending_work_units: 2,
        sessions: { 'session1': { status: 'active' } },
      };
      mockClient.workspaces.deriverStatus.mockResolvedValue(mockStatus);

      const status = await session.getDeriverStatus();

      expect(status).toEqual({
        totalWorkUnits: 10,
        completedWorkUnits: 5,
        inProgressWorkUnits: 3,
        pendingWorkUnits: 2,
        sessions: { 'session1': { status: 'active' } },
      });
      expect(mockClient.workspaces.deriverStatus).toHaveBeenCalledWith('test-workspace', { session_id: 'test-session' });
    });

    it('should return deriver status with options', async () => {
      const mockStatus = {
        total_work_units: 5,
        completed_work_units: 3,
        in_progress_work_units: 1,
        pending_work_units: 1,
      };
      mockClient.workspaces.deriverStatus.mockResolvedValue(mockStatus);

      const status = await session.getDeriverStatus({
        observerId: 'observer1',
        senderId: 'sender1',
      });

      expect(status).toEqual({
        totalWorkUnits: 5,
        completedWorkUnits: 3,
        inProgressWorkUnits: 1,
        pendingWorkUnits: 1,
        sessions: undefined,
      });
      expect(mockClient.workspaces.deriverStatus).toHaveBeenCalledWith('test-workspace', {
        observer_id: 'observer1',
        sender_id: 'sender1',
        session_id: 'test-session',
      });
    });
  });

  describe('pollDeriverStatus', () => {
    it('should poll until processing is complete', async () => {
      const mockStatusComplete = {
        total_work_units: 5,
        completed_work_units: 5,
        in_progress_work_units: 0,
        pending_work_units: 0,
      };
      mockClient.workspaces.deriverStatus.mockResolvedValue(mockStatusComplete);

      const status = await session.pollDeriverStatus();

      expect(status).toEqual({
        totalWorkUnits: 5,
        completedWorkUnits: 5,
        inProgressWorkUnits: 0,
        pendingWorkUnits: 0,
        sessions: undefined,
      });

      expect(mockClient.workspaces.deriverStatus).toHaveBeenCalledWith(
        'test-workspace',
        { session_id: 'test-session' }
      );
    });

    it('should timeout if processing takes too long', async () => {
      const mockStatusPending = {
        total_work_units: 5,
        completed_work_units: 2,
        in_progress_work_units: 2,
        pending_work_units: 1,
      };
      mockClient.workspaces.deriverStatus.mockResolvedValue(mockStatusPending);

      await expect(session.pollDeriverStatus({ timeoutMs: 100 })).rejects.toThrow();
    });
  })
});
