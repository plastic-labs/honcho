import { Peer } from '../src/peer';
import { SessionContext, Summary } from '../src/session_context';

/**
 * Helper function to create a proper Message object for testing
 */
function createTestMessage(id: string, content: string, peer_id: string, additionalProps: any = {}): any {
  return {
    id,
    content,
    peer_id,
    created_at: new Date().toISOString(),
    session_id: 'test-session',
    token_count: 0,
    workspace_id: 'test-workspace',
    ...additionalProps
  };
}

/**
 * Helper function to create a test Summary object
 */
function createTestSummary(content: string): Summary {
  return new Summary({
    content,
    message_id: '1',
    summary_type: 'short',
    created_at: new Date().toISOString(),
    token_count: content.length
  });
}

describe('SessionContext', () => {
  let sessionContext: SessionContext;
  let mockMessages: any[];

  beforeEach(() => {
    mockMessages = [
      createTestMessage('msg1', 'Hello', 'assistant'),
      createTestMessage('msg2', 'Hi there', 'user'),
      createTestMessage('msg3', 'How are you?', 'user'),
      createTestMessage('msg4', 'I am doing well, thank you!', 'assistant'),
    ];

    sessionContext = new SessionContext('test-session', mockMessages, null);
  });

  describe('constructor', () => {
    it('should initialize with all properties', () => {
      expect(sessionContext.sessionId).toBe('test-session');
      expect(sessionContext.messages).toEqual(mockMessages);
      expect(sessionContext.summary).toBe(null);
    });

    it('should initialize with null summary when not provided', () => {
      const context = new SessionContext('session-id', mockMessages);

      expect(context.sessionId).toBe('session-id');
      expect(context.messages).toEqual(mockMessages);
      expect(context.summary).toBe(null);
    });

    it('should handle empty messages array', () => {
      const summary = createTestSummary('No messages');
      const context = new SessionContext('session-id', [], summary);

      expect(context.sessionId).toBe('session-id');
      expect(context.messages).toEqual([]);
      expect(context.summary).toBe(summary);
    });

    it('should handle null/undefined summary', () => {
      const context1 = new SessionContext('session-id', mockMessages, undefined as any);
      const context2 = new SessionContext('session-id', mockMessages, null);

      expect(context1.summary).toBe(null);
      expect(context2.summary).toBe(null);
    });
  });

  describe('toOpenAI', () => {
    it('should convert messages to OpenAI format with string assistant', () => {
      const openAIMessages = sessionContext.toOpenAI('assistant');

      expect(openAIMessages).toEqual([
        { role: 'assistant', content: 'Hello', name: 'assistant' },
        { role: 'user', content: 'Hi there', name: 'user' },
        { role: 'user', content: 'How are you?', name: 'user' },
        { role: 'assistant', content: 'I am doing well, thank you!', name: 'assistant' },
      ]);
    });

    it('should convert messages to OpenAI format with Peer object', () => {
      const mockClient = {} as any;
      const assistantPeer = new Peer('assistant', 'test-workspace', mockClient);

      const openAIMessages = sessionContext.toOpenAI(assistantPeer);

      expect(openAIMessages).toEqual([
        { role: 'assistant', content: 'Hello', name: 'assistant' },
        { role: 'user', content: 'Hi there', name: 'user' },
        { role: 'user', content: 'How are you?', name: 'user' },
        { role: 'assistant', content: 'I am doing well, thank you!', name: 'assistant' },
      ]);
    });

    it('should handle messages where assistant is different peer', () => {
      const openAIMessages = sessionContext.toOpenAI('different-assistant');

      expect(openAIMessages).toEqual([
        { role: 'user', content: 'Hello', name: 'assistant' },
        { role: 'user', content: 'Hi there', name: 'user' },
        { role: 'user', content: 'How are you?', name: 'user' },
        { role: 'user', content: 'I am doing well, thank you!', name: 'assistant' },
      ]);
    });

    it('should handle empty messages array', () => {
      const emptyContext = new SessionContext('session-id', []);
      const openAIMessages = emptyContext.toOpenAI('assistant');

      expect(openAIMessages).toEqual([]);
    });

    it('should include summary message when summary exists', () => {
      const summary = createTestSummary('This is a summary');
      const contextWithSummary = new SessionContext('test-session', mockMessages, summary);
      const openAIMessages = contextWithSummary.toOpenAI('assistant');

      expect(openAIMessages).toEqual([
        { role: 'system', content: '<summary>This is a summary</summary>' },
        { role: 'assistant', content: 'Hello', name: 'assistant' },
        { role: 'user', content: 'Hi there', name: 'user' },
        { role: 'user', content: 'How are you?', name: 'user' },
        { role: 'assistant', content: 'I am doing well, thank you!', name: 'assistant' },
      ]);
    });

    it('should handle messages with missing peer_id', () => {
      const messagesWithMissingPeer = [
        createTestMessage('msg1', 'Hello', 'assistant'),
        createTestMessage('msg2', 'No peer', ''), // missing peer_id
        createTestMessage('msg3', 'Another message', ''), // null peer_id
      ];
      const context = new SessionContext('test', messagesWithMissingPeer);

      const openAIMessages = context.toOpenAI('assistant');

      expect(openAIMessages).toEqual([
        { role: 'assistant', content: 'Hello', name: 'assistant' },
        { role: 'user', content: 'No peer', name: '' },
        { role: 'user', content: 'Another message', name: '' },
      ]);
    });

    it('should handle complex message content', () => {
      const complexMessages = [
        createTestMessage('msg1', 'Message with\nnewlines and special chars!@#$%', 'assistant'),
        createTestMessage('msg2', '', 'user'), // empty content
        createTestMessage('msg3', '   whitespace   ', 'assistant'),
      ];
      const context = new SessionContext('test', complexMessages);

      const openAIMessages = context.toOpenAI('assistant');

      expect(openAIMessages).toEqual([
        { role: 'assistant', content: 'Message with\nnewlines and special chars!@#$%', name: 'assistant' },
        { role: 'user', content: '', name: 'user' },
        { role: 'assistant', content: '   whitespace   ', name: 'assistant' },
      ]);
    });
  });

  describe('toAnthropic', () => {
    it('should convert messages to Anthropic format with string assistant', () => {
      const anthropicMessages = sessionContext.toAnthropic('assistant');

      expect(anthropicMessages).toEqual([
        { role: 'assistant', content: 'Hello' },
        { role: 'user', content: 'user: Hi there' },
        { role: 'user', content: 'user: How are you?' },
        { role: 'assistant', content: 'I am doing well, thank you!' },
      ]);
    });

    it('should convert messages to Anthropic format with Peer object', () => {
      const mockClient = {} as any;
      const assistantPeer = new Peer('assistant', 'test-workspace', mockClient);

      const anthropicMessages = sessionContext.toAnthropic(assistantPeer);

      expect(anthropicMessages).toEqual([
        { role: 'assistant', content: 'Hello' },
        { role: 'user', content: 'user: Hi there' },
        { role: 'user', content: 'user: How are you?' },
        { role: 'assistant', content: 'I am doing well, thank you!' },
      ]);
    });

    it('should handle messages where assistant is different peer', () => {
      const anthropicMessages = sessionContext.toAnthropic('different-assistant');

      expect(anthropicMessages).toEqual([
        { role: 'user', content: 'assistant: Hello' },
        { role: 'user', content: 'user: Hi there' },
        { role: 'user', content: 'user: How are you?' },
        { role: 'user', content: 'assistant: I am doing well, thank you!' },
      ]);
    });

    it('should handle empty messages array', () => {
      const emptyContext = new SessionContext('session-id', []);
      const anthropicMessages = emptyContext.toAnthropic('assistant');

      expect(anthropicMessages).toEqual([]);
    });

    it('should include summary message when summary exists', () => {
      const summary = createTestSummary('This is a summary');
      const contextWithSummary = new SessionContext('test-session', mockMessages, summary);
      const anthropicMessages = contextWithSummary.toAnthropic('assistant');

      expect(anthropicMessages).toEqual([
        { role: 'user', content: '<summary>This is a summary</summary>' },
        { role: 'assistant', content: 'Hello' },
        { role: 'user', content: 'user: Hi there' },
        { role: 'user', content: 'user: How are you?' },
        { role: 'assistant', content: 'I am doing well, thank you!' },
      ]);
    });

    it('should handle messages with missing peer_id', () => {
      const messagesWithMissingPeer = [
        createTestMessage('msg1', 'Hello', 'assistant'),
        createTestMessage('msg2', 'No peer', ''), // missing peer_id
        createTestMessage('msg3', 'Another message', ''), // undefined peer_id
      ];
      const context = new SessionContext('test', messagesWithMissingPeer);

      const anthropicMessages = context.toAnthropic('assistant');

      expect(anthropicMessages).toEqual([
        { role: 'assistant', content: 'Hello' },
        { role: 'user', content: ': No peer' },
        { role: 'user', content: ': Another message' },
      ]);
    });
  });

  describe('length getter', () => {
    it('should return correct message count', () => {
      expect(sessionContext.length).toBe(4);
    });

    it('should return zero for empty messages', () => {
      const emptyContext = new SessionContext('session-id', []);
      expect(emptyContext.length).toBe(0);
    });

    it('should return correct count for single message', () => {
      const singleMessageContext = new SessionContext('session-id', [mockMessages[0]]);
      expect(singleMessageContext.length).toBe(1);
    });
  });

  describe('toString', () => {
    it('should return correct string representation', () => {
      const result = sessionContext.toString();
      expect(result).toBe('SessionContext(messages=4, summary=none)');
    });

    it('should handle empty messages', () => {
      const emptyContext = new SessionContext('session-id', []);
      const result = emptyContext.toString();
      expect(result).toBe('SessionContext(messages=0, summary=none)');
    });

    it('should handle large number of messages', () => {
      const manyMessages = Array.from({ length: 1000 }, (_, i) =>
        createTestMessage(`msg${i}`, `Message ${i}`, i % 2 === 0 ? 'assistant' : 'user')
      );
      const context = new SessionContext('session-id', manyMessages);

      const result = context.toString();
      expect(result).toBe('SessionContext(messages=1000, summary=none)');
    });
  });

  describe('edge cases and error handling', () => {
    it('should handle messages with null content', () => {
      const messagesWithNullContent = [
        createTestMessage('msg1', null as any, 'assistant'),
        createTestMessage('msg2', undefined as any, 'user'),
      ];
      const context = new SessionContext('test', messagesWithNullContent);

      const openAIMessages = context.toOpenAI('assistant');

      expect(openAIMessages).toEqual([
        { role: 'assistant', content: null, name: 'assistant' },
        { role: 'user', content: undefined, name: 'user' },
      ]);
    });

    it('should handle messages with non-string content', () => {
      const messagesWithNonStringContent = [
        createTestMessage('msg1', 123 as any, 'assistant'),
        createTestMessage('msg2', { text: 'object content' } as any, 'user'),
        createTestMessage('msg3', true as any, 'assistant'),
      ];
      const context = new SessionContext('test', messagesWithNonStringContent);

      const openAIMessages = context.toOpenAI('assistant');

      expect(openAIMessages).toEqual([
        { role: 'assistant', content: 123, name: 'assistant' },
        { role: 'user', content: { text: 'object content' }, name: 'user' },
        { role: 'assistant', content: true, name: 'assistant' },
      ]);
    });

    it('should handle very long session IDs and summaries', () => {
      const longSessionId = 'x'.repeat(1000);
      const longSummaryText = 'Very long summary that goes on and on...'.repeat(100);
      const longSummary = createTestSummary(longSummaryText);
      const context = new SessionContext(longSessionId, mockMessages, longSummary);

      expect(context.sessionId).toBe(longSessionId);
      expect(context.summary).toBe(longSummary);
      expect(context.summary?.content).toBe(longSummaryText);
      expect(context.length).toBe(5); // 4 messages + 1 summary
    });

    it('should handle messages with additional properties', () => {
      const messagesWithExtraProps = [
        createTestMessage('msg1', 'Hello', 'assistant', {
          timestamp: '2023-01-01T00:00:00Z',
          metadata: { important: true },
          extra_field: 'extra_value'
        }),
      ];
      const context = new SessionContext('test', messagesWithExtraProps);

      const openAIMessages = context.toOpenAI('assistant');

      expect(openAIMessages).toEqual([
        { role: 'assistant', content: 'Hello', name: 'assistant' },
      ]);
    });

    it('should handle case-sensitive peer names', () => {
      const caseMessages = [
        createTestMessage('msg1', 'Hello', 'Assistant'),
        createTestMessage('msg2', 'Hi', 'ASSISTANT'),
        createTestMessage('msg3', 'Hey', 'assistant'),
      ];
      const context = new SessionContext('test', caseMessages);

      const openAIMessages = context.toOpenAI('assistant');

      expect(openAIMessages).toEqual([
        { role: 'user', content: 'Hello', name: 'Assistant' }, // 'Assistant' != 'assistant'
        { role: 'user', content: 'Hi', name: 'ASSISTANT' }, // 'ASSISTANT' != 'assistant'
        { role: 'assistant', content: 'Hey', name: 'assistant' }, // exact match
      ]);
    });

    it('should handle messages without id field', () => {
      const messagesWithoutId = [
        createTestMessage('', 'Message without ID', 'assistant'),
        createTestMessage('', 'Another message', 'user'),
      ];
      const context = new SessionContext('test', messagesWithoutId);

      expect(context.length).toBe(2);
      expect(context.toOpenAI('assistant')).toEqual([
        { role: 'assistant', content: 'Message without ID', name: 'assistant' },
        { role: 'user', content: 'Another message', name: 'user' },
      ]);
    });
  });
});
