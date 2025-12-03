import { Honcho } from '../src/client';
import { Peer } from '../src/peer';
import { Session } from '../src/session';

// Mock the @honcho-ai/core module
jest.mock('@honcho-ai/core', () => {
  return jest.fn().mockImplementation(() => ({
    workspaces: {
      peers: {
        list: jest.fn(),
        getOrCreate: jest.fn(),
        update: jest.fn(),
      },
      sessions: {
        list: jest.fn(),
        getOrCreate: jest.fn(),
        update: jest.fn(),
      },
      getOrCreate: jest.fn(),
      update: jest.fn(),
    },
  }));
});

describe('Metadata and Configuration Caching', () => {
  let honcho: Honcho;
  let mockClient: any;

  beforeEach(() => {
    jest.clearAllMocks();

    honcho = new Honcho({
      workspaceId: 'test-workspace',
      apiKey: 'test-key',
      environment: 'local',
    });

    mockClient = (honcho as any)._client;
  });

  describe('Workspace Metadata Caching', () => {
    it('should initialize with undefined metadata', () => {
      expect(honcho.metadata).toBeUndefined();
    });

    it('should cache metadata after getMetadata call', async () => {
      const mockWorkspace = {
        id: 'test-workspace',
        metadata: { theme: 'dark', version: '1.0' },
      };
      mockClient.workspaces.getOrCreate.mockResolvedValue(mockWorkspace);

      const metadata = await honcho.getMetadata();

      expect(metadata).toEqual({ theme: 'dark', version: '1.0' });
      expect(honcho.metadata).toEqual({ theme: 'dark', version: '1.0' });
    });

    it('should cache empty object when metadata is null', async () => {
      const mockWorkspace = {
        id: 'test-workspace',
        metadata: null,
      };
      mockClient.workspaces.getOrCreate.mockResolvedValue(mockWorkspace);

      const metadata = await honcho.getMetadata();

      expect(metadata).toEqual({});
      expect(honcho.metadata).toEqual({});
    });

    it('should update cached metadata after setMetadata call', async () => {
      mockClient.workspaces.update.mockResolvedValue({});

      const newMetadata = { theme: 'light', version: '2.0' };
      await honcho.setMetadata(newMetadata);

      expect(honcho.metadata).toEqual(newMetadata);
    });

    it('should maintain cached value across multiple calls', async () => {
      const mockWorkspace = {
        id: 'test-workspace',
        metadata: { count: 1 },
      };
      mockClient.workspaces.getOrCreate.mockResolvedValue(mockWorkspace);

      await honcho.getMetadata();
      expect(honcho.metadata).toEqual({ count: 1 });

      // Update cache
      mockClient.workspaces.update.mockResolvedValue({});
      await honcho.setMetadata({ count: 2 });
      expect(honcho.metadata).toEqual({ count: 2 });

      // Verify cache persists
      expect(honcho.metadata).toEqual({ count: 2 });
    });
  });

  describe('Peer Metadata and Configuration Caching', () => {
    describe('Peer Constructor with metadata/config', () => {
      it('should initialize peer with provided metadata and config', async () => {
        const metadata = { name: 'Test Peer', role: 'assistant' };
        const config = { observe_me: false };

        mockClient.workspaces.peers.getOrCreate.mockResolvedValue({
          id: 'peer1',
          metadata: metadata,
          configuration: config,
        });

        const peer = await honcho.peer('peer1', { metadata, config });

        expect(peer.metadata).toEqual(metadata);
        expect(peer.configuration).toEqual(config);
        expect(mockClient.workspaces.peers.getOrCreate).toHaveBeenCalledWith(
          'test-workspace',
          { id: 'peer1', metadata, configuration: config }
        );
      });

      it('should initialize peer without metadata/config', async () => {
        const peer = await honcho.peer('peer1');

        expect(peer.metadata).toBeUndefined();
        expect(peer.configuration).toBeUndefined();
        expect(mockClient.workspaces.peers.getOrCreate).not.toHaveBeenCalled();
      });
    });

    describe('Peer Metadata Caching', () => {
      let peer: Peer;

      beforeEach(() => {
        peer = new Peer('test-peer', 'test-workspace', mockClient);
      });

      it('should cache metadata after getMetadata call', async () => {
        const mockPeer = {
          id: 'test-peer',
          metadata: { name: 'Alice', role: 'user' },
        };
        mockClient.workspaces.peers.getOrCreate.mockResolvedValue(mockPeer);

        const metadata = await peer.getMetadata();

        expect(metadata).toEqual({ name: 'Alice', role: 'user' });
        expect(peer.metadata).toEqual({ name: 'Alice', role: 'user' });
      });

      it('should cache empty object when metadata is null', async () => {
        const mockPeer = {
          id: 'test-peer',
          metadata: null,
        };
        mockClient.workspaces.peers.getOrCreate.mockResolvedValue(mockPeer);

        const metadata = await peer.getMetadata();

        expect(metadata).toEqual({});
        expect(peer.metadata).toEqual({});
      });

      it('should update cached metadata after setMetadata call', async () => {
        mockClient.workspaces.peers.update.mockResolvedValue({});

        const newMetadata = { name: 'Bob', role: 'admin' };
        await peer.setMetadata(newMetadata);

        expect(peer.metadata).toEqual(newMetadata);
      });

      it('should maintain cached value across operations', async () => {
        mockClient.workspaces.peers.getOrCreate.mockResolvedValue({
          id: 'test-peer',
          metadata: { score: 100 },
        });
        mockClient.workspaces.peers.update.mockResolvedValue({});

        // Get initial metadata
        await peer.getMetadata();
        expect(peer.metadata).toEqual({ score: 100 });

        // Update metadata
        await peer.setMetadata({ score: 200 });
        expect(peer.metadata).toEqual({ score: 200 });

        // Verify cache persists
        expect(peer.metadata).toEqual({ score: 200 });
      });
    });

    describe('Peer Configuration Caching', () => {
      let peer: Peer;

      beforeEach(() => {
        peer = new Peer('test-peer', 'test-workspace', mockClient);
      });

      it('should cache configuration after getConfig call', async () => {
        const mockPeer = {
          id: 'test-peer',
          configuration: { observe_me: true, observe_others: false },
        };
        mockClient.workspaces.peers.getOrCreate.mockResolvedValue(mockPeer);

        const config = await peer.getConfig();

        expect(config).toEqual({ observe_me: true, observe_others: false });
        expect(peer.configuration).toEqual({ observe_me: true, observe_others: false });
      });

      it('should cache empty object when configuration is null', async () => {
        const mockPeer = {
          id: 'test-peer',
          configuration: null,
        };
        mockClient.workspaces.peers.getOrCreate.mockResolvedValue(mockPeer);

        const config = await peer.getConfig();

        expect(config).toEqual({});
        expect(peer.configuration).toEqual({});
      });

      it('should update cached configuration after setConfig call', async () => {
        mockClient.workspaces.peers.update.mockResolvedValue({});

        const newConfig = { observe_me: false, observe_others: true };
        await peer.setConfig(newConfig);

        expect(peer.configuration).toEqual(newConfig);
      });

      it('should support deprecated getPeerConfig method', async () => {
        const mockPeer = {
          id: 'test-peer',
          configuration: { observe_me: true },
        };
        mockClient.workspaces.peers.getOrCreate.mockResolvedValue(mockPeer);

        const config = await peer.getPeerConfig();

        expect(config).toEqual({ observe_me: true });
        expect(peer.configuration).toEqual({ observe_me: true });
      });

      it('should support deprecated setPeerConfig method', async () => {
        mockClient.workspaces.peers.update.mockResolvedValue({});

        const newConfig = { observe_me: false };
        await peer.setPeerConfig(newConfig);

        expect(peer.configuration).toEqual(newConfig);
      });
    });

    describe('Peer List with Cached Data', () => {
      it('should populate metadata and config when listing peers', async () => {
        const mockPeersData = {
          items: [
            {
              id: 'peer1',
              metadata: { name: 'Alice' },
              configuration: { observe_me: true },
            },
            {
              id: 'peer2',
              metadata: { name: 'Bob' },
              configuration: { observe_me: false },
            },
          ],
          total: 2,
          size: 2,
          hasNextPage: false,
        };
        mockClient.workspaces.peers.list.mockResolvedValue(mockPeersData);

        const peersPage = await honcho.getPeers();
        const peers = peersPage.items;

        expect(peers[0].metadata).toEqual({ name: 'Alice' });
        expect(peers[0].configuration).toEqual({ observe_me: true });
        expect(peers[1].metadata).toEqual({ name: 'Bob' });
        expect(peers[1].configuration).toEqual({ observe_me: false });
      });

      it('should handle null metadata and config in peer list', async () => {
        const mockPeersData = {
          items: [
            {
              id: 'peer1',
              metadata: null,
              configuration: null,
            },
          ],
          total: 1,
          size: 1,
          hasNextPage: false,
        };
        mockClient.workspaces.peers.list.mockResolvedValue(mockPeersData);

        const peersPage = await honcho.getPeers();
        const peers = peersPage.items;

        expect(peers[0].metadata).toBeUndefined();
        expect(peers[0].configuration).toBeUndefined();
      });
    });
  });

  describe('Session Metadata and Configuration Caching', () => {
    describe('Session Constructor with metadata/config', () => {
      it('should initialize session with provided metadata and config', async () => {
        const metadata = { title: 'Test Session', tags: ['important'] };
        const config = { anonymous: false };

        mockClient.workspaces.sessions.getOrCreate.mockResolvedValue({
          id: 'session1',
          metadata: metadata,
          configuration: config,
        });

        const session = await honcho.session('session1', { metadata, config });

        expect(session.metadata).toEqual(metadata);
        expect(session.configuration).toEqual(config);
        expect(mockClient.workspaces.sessions.getOrCreate).toHaveBeenCalledWith(
          'test-workspace',
          { id: 'session1', metadata, configuration: config }
        );
      });

      it('should initialize session without metadata/config', async () => {
        const session = await honcho.session('session1');

        expect(session.metadata).toBeUndefined();
        expect(session.configuration).toBeUndefined();
        expect(mockClient.workspaces.sessions.getOrCreate).not.toHaveBeenCalled();
      });
    });

    describe('Session Metadata Caching', () => {
      let session: Session;

      beforeEach(() => {
        session = new Session('test-session', 'test-workspace', mockClient);
      });

      it('should cache metadata after getMetadata call', async () => {
        const mockSession = {
          id: 'test-session',
          metadata: { title: 'Chat Session', active: true },
        };
        mockClient.workspaces.sessions.getOrCreate.mockResolvedValue(mockSession);

        const metadata = await session.getMetadata();

        expect(metadata).toEqual({ title: 'Chat Session', active: true });
        expect(session.metadata).toEqual({ title: 'Chat Session', active: true });
      });

      it('should cache empty object when metadata is null', async () => {
        const mockSession = {
          id: 'test-session',
          metadata: null,
        };
        mockClient.workspaces.sessions.getOrCreate.mockResolvedValue(mockSession);

        const metadata = await session.getMetadata();

        expect(metadata).toEqual({});
        expect(session.metadata).toEqual({});
      });

      it('should update cached metadata after setMetadata call', async () => {
        mockClient.workspaces.sessions.update.mockResolvedValue({});

        const newMetadata = { title: 'Updated Session', active: false };
        await session.setMetadata(newMetadata);

        expect(session.metadata).toEqual(newMetadata);
      });
    });

    describe('Session Configuration Caching', () => {
      let session: Session;

      beforeEach(() => {
        session = new Session('test-session', 'test-workspace', mockClient);
      });

      it('should cache configuration after getConfig call', async () => {
        const mockSession = {
          id: 'test-session',
          configuration: { anonymous: true, summarize: false },
        };
        mockClient.workspaces.sessions.getOrCreate.mockResolvedValue(mockSession);

        const config = await session.getConfig();

        expect(config).toEqual({ anonymous: true, summarize: false });
        expect(session.configuration).toEqual({ anonymous: true, summarize: false });
      });

      it('should cache empty object when configuration is null', async () => {
        const mockSession = {
          id: 'test-session',
          configuration: null,
        };
        mockClient.workspaces.sessions.getOrCreate.mockResolvedValue(mockSession);

        const config = await session.getConfig();

        expect(config).toEqual({});
        expect(session.configuration).toEqual({});
      });

      it('should update cached configuration after setConfig call', async () => {
        mockClient.workspaces.sessions.update.mockResolvedValue({});

        const newConfig = { anonymous: false, summarize: true };
        await session.setConfig(newConfig);

        expect(session.configuration).toEqual(newConfig);
      });
    });

    describe('Session List with Cached Data', () => {
      it('should populate metadata and config when listing sessions', async () => {
        const mockSessionsData = {
          items: [
            {
              id: 'session1',
              metadata: { title: 'Session 1' },
              configuration: { anonymous: true },
            },
            {
              id: 'session2',
              metadata: { title: 'Session 2' },
              configuration: { anonymous: false },
            },
          ],
          total: 2,
          size: 2,
          hasNextPage: false,
        };
        mockClient.workspaces.sessions.list.mockResolvedValue(mockSessionsData);

        const sessionsPage = await honcho.getSessions();
        const sessions = sessionsPage.items;

        expect(sessions[0].metadata).toEqual({ title: 'Session 1' });
        expect(sessions[0].configuration).toEqual({ anonymous: true });
        expect(sessions[1].metadata).toEqual({ title: 'Session 2' });
        expect(sessions[1].configuration).toEqual({ anonymous: false });
      });

      it('should handle null metadata and config in session list', async () => {
        const mockSessionsData = {
          items: [
            {
              id: 'session1',
              metadata: null,
              configuration: null,
            },
          ],
          total: 1,
          size: 1,
          hasNextPage: false,
        };
        mockClient.workspaces.sessions.list.mockResolvedValue(mockSessionsData);

        const sessionsPage = await honcho.getSessions();
        const sessions = sessionsPage.items;

        expect(sessions[0].metadata).toBeUndefined();
        expect(sessions[0].configuration).toBeUndefined();
      });
    });
  });

  describe('Integration: Combined Metadata and Configuration Operations', () => {
    it('should cache both metadata and config for peers independently', async () => {
      const peer = new Peer('test-peer', 'test-workspace', mockClient);

      // Set up mocks
      mockClient.workspaces.peers.getOrCreate.mockResolvedValue({
        id: 'test-peer',
        metadata: { name: 'Test' },
        configuration: { observe_me: true },
      });
      mockClient.workspaces.peers.update.mockResolvedValue({});

      // Get both metadata and config
      await peer.getMetadata();
      await peer.getConfig();

      expect(peer.metadata).toEqual({ name: 'Test' });
      expect(peer.configuration).toEqual({ observe_me: true });

      // Update metadata only
      await peer.setMetadata({ name: 'Updated' });

      expect(peer.metadata).toEqual({ name: 'Updated' });
      expect(peer.configuration).toEqual({ observe_me: true }); // Should remain unchanged

      // Update config only
      await peer.setConfig({ observe_me: false });

      expect(peer.metadata).toEqual({ name: 'Updated' }); // Should remain unchanged
      expect(peer.configuration).toEqual({ observe_me: false });
    });

    it('should cache both metadata and config for sessions independently', async () => {
      const session = new Session('test-session', 'test-workspace', mockClient);

      // Set up mocks
      mockClient.workspaces.sessions.getOrCreate.mockResolvedValue({
        id: 'test-session',
        metadata: { title: 'Test' },
        configuration: { anonymous: true },
      });
      mockClient.workspaces.sessions.update.mockResolvedValue({});

      // Get both metadata and config
      await session.getMetadata();
      await session.getConfig();

      expect(session.metadata).toEqual({ title: 'Test' });
      expect(session.configuration).toEqual({ anonymous: true });

      // Update metadata only
      await session.setMetadata({ title: 'Updated' });

      expect(session.metadata).toEqual({ title: 'Updated' });
      expect(session.configuration).toEqual({ anonymous: true }); // Should remain unchanged

      // Update config only
      await session.setConfig({ anonymous: false });

      expect(session.metadata).toEqual({ title: 'Updated' }); // Should remain unchanged
      expect(session.configuration).toEqual({ anonymous: false });
    });

    it('should reduce API calls by using cached values', async () => {
      const peer = new Peer('test-peer', 'test-workspace', mockClient);

      // Initial fetch
      mockClient.workspaces.peers.getOrCreate.mockResolvedValue({
        id: 'test-peer',
        metadata: { name: 'Test' },
      });

      await peer.getMetadata();
      expect(mockClient.workspaces.peers.getOrCreate).toHaveBeenCalledTimes(1);

      // Access cached value directly (without API call)
      const cachedMetadata = peer.metadata;
      expect(cachedMetadata).toEqual({ name: 'Test' });
      expect(mockClient.workspaces.peers.getOrCreate).toHaveBeenCalledTimes(1); // Still only 1 call
    });
  });
});
