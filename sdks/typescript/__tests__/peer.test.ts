import { Peer } from '../src/peer';
import { Session } from '../src/session';
import { Page } from '../src/pagination';
import { Honcho } from '../src/client';

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

    peer = new Peer('test-peer', honcho);
    mockClient = (honcho as any)._client;
  });

  describe('constructor', () => {
    it('should initialize with correct properties', () => {
      const newPeer = new Peer('peer-id', honcho);

      expect(newPeer.id).toBe('peer-id');
      expect(newPeer['_honcho']).toBe(honcho);
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
        { query: 'Hello', stream: undefined, target: undefined, session_id: undefined }
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

    it('should handle chat with streaming option', async () => {
      const mockResponse = { content: 'Streamed response' };
      mockClient.workspaces.peers.chat.mockResolvedValue(mockResponse);

      await peer.chat('Hello', { stream: true });

      expect(mockClient.workspaces.peers.chat).toHaveBeenCalledWith(
        'test-workspace',
        'test-peer',
        { query: 'Hello', stream: true, target: undefined, session_id: undefined }
      );
    });

    it('should handle chat with target peer', async () => {
      const targetPeer = new Peer('target-peer', honcho);
      const mockResponse = { content: 'Targeted response' };
      mockClient.workspaces.peers.chat.mockResolvedValue(mockResponse);

      await peer.chat('Hello', { target: targetPeer });

      expect(mockClient.workspaces.peers.chat).toHaveBeenCalledWith(
        'test-workspace',
        'test-peer',
        { query: 'Hello', stream: undefined, target: 'target-peer', session_id: undefined }
      );
    });

    it('should handle chat with target as string', async () => {
      const mockResponse = { content: 'Targeted response' };
      mockClient.workspaces.peers.chat.mockResolvedValue(mockResponse);

      await peer.chat('Hello', { target: 'string-target' });

      expect(mockClient.workspaces.peers.chat).toHaveBeenCalledWith(
        'test-workspace',
        'test-peer',
        { query: 'Hello', stream: undefined, target: 'string-target', session_id: undefined }
      );
    });

    it('should handle chat with session ID', async () => {
      const mockResponse = { content: 'Session-specific response' };
      mockClient.workspaces.peers.chat.mockResolvedValue(mockResponse);

      await peer.chat('Hello', { sessionId: 'session-123' });

      expect(mockClient.workspaces.peers.chat).toHaveBeenCalledWith(
        'test-workspace',
        'test-peer',
        { query: 'Hello', stream: undefined, target: undefined, session_id: 'session-123' }
      );
    });

    it('should handle all options together', async () => {
      const targetPeer = new Peer('target-peer', honcho);
      const mockResponse = { content: 'Full options response' };
      mockClient.workspaces.peers.chat.mockResolvedValue(mockResponse);

      await peer.chat('Hello', {
        stream: true,
        target: targetPeer,
        sessionId: 'session-456'
      });

      expect(mockClient.workspaces.peers.chat).toHaveBeenCalledWith(
        'test-workspace',
        'test-peer',
        { query: 'Hello', stream: true, target: 'target-peer', session_id: 'session-456' }
      );
    });

    it('should handle API errors', async () => {
      mockClient.workspaces.peers.chat.mockRejectedValue(new Error('Chat failed'));

      await expect(peer.chat('Hello')).rejects.toThrow('Chat failed');
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
        'test-peer',
        'test-workspace'
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

      await expect(peer.getSessions()).rejects.toThrow('Failed to get sessions');
    });
  });

  describe('message', () => {
    it('should create message object without metadata', () => {
      const message = peer.message('Test content');

      expect(message).toEqual({
        peerId: 'test-peer',
        content: 'Test content',
        metadata: undefined,
      });
    });

    it('should create message object with metadata', () => {
      const metadata = { importance: 'high', category: 'greeting' };
      const message = peer.message('Hello there', { metadata });

      expect(message).toEqual({
        peerId: 'test-peer',
        content: 'Hello there',
        metadata: { importance: 'high', category: 'greeting' },
      });
    });

    it('should handle empty content', () => {
      const message = peer.message('');

      expect(message).toEqual({
        peerId: 'test-peer',
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

      await expect(peer.getMetadata()).rejects.toThrow('Peer not found');
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

      await expect(peer.setMetadata({ key: 'value' })).rejects.toThrow('Update failed');
    });
  });

  describe('search', () => {
    it('should search peer messages and return Page', async () => {
      const mockSearchResults = {
        items: [
          { id: 'msg1', content: 'Hello world', peer_id: 'test-peer' },
          { id: 'msg2', content: 'Hello there', peer_id: 'test-peer' },
        ],
        total: 2,
        size: 2,
        hasNextPage: false,
      };
      mockClient.workspaces.peers.search.mockResolvedValue(mockSearchResults);

      const results = await peer.search('hello');

      expect(results).toBeInstanceOf(Page);
      expect(mockClient.workspaces.peers.search).toHaveBeenCalledWith(
        'test-workspace',
        'test-peer',
        { body: 'hello' }
      );
    });

    it('should handle empty search results', async () => {
      const mockSearchResults = {
        items: [],
        total: 0,
        size: 0,
        hasNextPage: false,
      };
      mockClient.workspaces.peers.search.mockResolvedValue(mockSearchResults);

      const results = await peer.search('nonexistent');

      expect(results).toBeInstanceOf(Page);
    });

    it('should throw error for empty query', async () => {
      await expect(peer.search('')).rejects.toThrow('Search query must be a non-empty string');
      await expect(peer.search('   ')).rejects.toThrow('Search query must be a non-empty string');
    });

    it('should throw error for non-string query', async () => {
      await expect(peer.search(null as any)).rejects.toThrow('Search query must be a non-empty string');
      await expect(peer.search(undefined as any)).rejects.toThrow('Search query must be a non-empty string');
      await expect(peer.search(123 as any)).rejects.toThrow('Search query must be a non-empty string');
    });

    it('should handle complex search queries', async () => {
      const mockSearchResults = {
        items: [],
        total: 0,
        size: 0,
        hasNextPage: false,
      };
      mockClient.workspaces.peers.search.mockResolvedValue(mockSearchResults);

      const complexQuery = 'complex query with "quotes" and special characters!@#$%';
      await peer.search(complexQuery);

      expect(mockClient.workspaces.peers.search).toHaveBeenCalledWith(
        'test-workspace',
        'test-peer',
        { body: complexQuery }
      );
    });

    it('should handle API errors', async () => {
      mockClient.workspaces.peers.search.mockRejectedValue(new Error('Search failed'));

      await expect(peer.search('test')).rejects.toThrow('Search failed');
    });
  });
}); 