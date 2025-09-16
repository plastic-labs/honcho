import { Message } from '../src/message';
import { Peer } from '../src/peer';
import { Session } from '../src/session';

/**
 * Helper function to create a test message object
 */
function createTestMessage(id: string, content: string, peer_id: string, metadata: Record<string, unknown> = {}): any {
    return {
        id,
        content,
        created_at: new Date().toISOString(),
        peer_id,
        session_id: 'test-session',
        token_count: 10,
        workspace_id: 'test-workspace',
        metadata
    };
}

describe('Message', () => {
    let mockClient: any;

    beforeEach(() => {
        // Create a comprehensive mock client for session operations
        mockClient = {
            workspaces: {
                sessions: {
                    messages: {
                        create: jest.fn().mockResolvedValue({}),
                        list: jest.fn().mockResolvedValue({
                            items: [{
                                id: 'test-message-id',
                                content: 'Test message content',
                                created_at: new Date().toISOString(),
                                peer_id: 'test-peer',
                                session_id: 'test-session',
                                token_count: 10,
                                workspace_id: 'test-workspace',
                                metadata: null
                            }],
                            total: 1,
                            size: 1,
                            hasNextPage: false
                        }),
                        update: jest.fn().mockResolvedValue({})
                    },
                    getOrCreate: jest.fn().mockResolvedValue({
                        id: 'test-session',
                        metadata: {}
                    })
                },
                peers: {
                    getOrCreate: jest.fn().mockResolvedValue({
                        id: 'test-peer',
                        metadata: {}
                    })
                }
            }
        };
    });

    describe('update', () => {
        it('should update message metadata', async () => {
            // Create a message instance using the helper function
            const messageData = createTestMessage('test-message-id', 'Test message content', 'test-peer', { original: 'value' });
            const messageInstance = Message.fromCore(messageData, mockClient);

            // Update the message metadata
            const newMetadata = { updated: 'new-value', status: 'modified' };
            const updatedMessage = await messageInstance.update(newMetadata);

            // Verify the API was called correctly
            expect(mockClient.workspaces.sessions.messages.update).toHaveBeenCalledWith(
                'test-workspace',
                'test-session',
                'test-message-id',
                { metadata: newMetadata }
            );

            // Verify the local metadata was updated
            expect(updatedMessage.metadata).toEqual(newMetadata);
            expect(updatedMessage).toBe(messageInstance);
        });

        it('should follow session workflow: create session, add message, validate no metadata, update metadata', async () => {
            // 1. Create session and peer
            const session = new Session('test-session', 'test-workspace', mockClient);
            const peer = new Peer('test-peer', 'test-workspace', mockClient);

            // 2. Add message to session
            const messageToAdd = peer.message('Test message content');
            await session.addMessages([messageToAdd]);

            // Verify the create API was called
            expect(mockClient.workspaces.sessions.messages.create).toHaveBeenCalledWith(
                'test-workspace',
                'test-session',
                {
                    messages: [{
                        peer_id: 'test-peer',
                        content: 'Test message content',
                        metadata: undefined
                    }]
                }
            );

            // 3. Get messages from session and validate no metadata
            const messagesPage = await session.getMessages();
            const messages = await messagesPage.data();
            expect(messages).toHaveLength(1);
            expect(messages[0].metadata).toEqual({}); // Should be empty object, not null

            // 4. Update message metadata
            const newMetadata = { foo: 'bar', status: 'updated' };
            await messages[0].update(newMetadata);

            // Verify the update API was called
            expect(mockClient.workspaces.sessions.messages.update).toHaveBeenCalledWith(
                'test-workspace',
                'test-session',
                'test-message-id',
                { metadata: newMetadata }
            );

            // 5. Verify the local metadata was updated
            expect(messages[0].metadata).toEqual(newMetadata);
        });
    });
});
