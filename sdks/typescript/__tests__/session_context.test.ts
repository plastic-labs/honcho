import { SessionContext } from '../src/session_context';
import { Peer } from '../src/peer';

describe('SessionContext', () => {
  let sessionContext: SessionContext;
  let mockMessages: any[];

  beforeEach(() => {
    mockMessages = [
      { id: 'msg1', content: 'Hello', peer_name: 'assistant' },
      { id: 'msg2', content: 'Hi there', peer_name: 'user' },
      { id: 'msg3', content: 'How are you?', peer_name: 'user' },
      { id: 'msg4', content: 'I am doing well, thank you!', peer_name: 'assistant' },
    ];

    sessionContext = new SessionContext('test-session', mockMessages, 'This is a summary');
  });

  describe('constructor', () => {
    it('should initialize with all properties', () => {
      expect(sessionContext.sessionId).toBe('test-session');
      expect(sessionContext.messages).toEqual(mockMessages);
      expect(sessionContext.summary).toBe('This is a summary');
    });

    it('should initialize with empty summary when not provided', () => {
      const context = new SessionContext('session-id', mockMessages);

      expect(context.sessionId).toBe('session-id');
      expect(context.messages).toEqual(mockMessages);
      expect(context.summary).toBe('');
    });

    it('should handle empty messages array', () => {
      const context = new SessionContext('session-id', [], 'No messages');

      expect(context.sessionId).toBe('session-id');
      expect(context.messages).toEqual([]);
      expect(context.summary).toBe('No messages');
    });

    it('should handle null/undefined summary', () => {
      const context1 = new SessionContext('session-id', mockMessages, undefined as any);
      const context2 = new SessionContext('session-id', mockMessages, null as any);

      expect(context1.summary).toBe('');
      expect(context2.summary).toBe('');
    });
  });

  describe('toOpenAI', () => {
    it('should convert messages to OpenAI format with string assistant', () => {
      const openAIMessages = sessionContext.toOpenAI('assistant');

      expect(openAIMessages).toEqual([
        { role: 'assistant', content: 'Hello' },
        { role: 'user', content: 'Hi there' },
        { role: 'user', content: 'How are you?' },
        { role: 'assistant', content: 'I am doing well, thank you!' },
      ]);
    });

    it('should convert messages to OpenAI format with Peer object', () => {
      const mockHoncho = {} as any;
      const assistantPeer = new Peer('assistant', mockHoncho);

      const openAIMessages = sessionContext.toOpenAI(assistantPeer);

      expect(openAIMessages).toEqual([
        { role: 'assistant', content: 'Hello' },
        { role: 'user', content: 'Hi there' },
        { role: 'user', content: 'How are you?' },
        { role: 'assistant', content: 'I am doing well, thank you!' },
      ]);
    });

    it('should handle messages where assistant is different peer', () => {
      const openAIMessages = sessionContext.toOpenAI('different-assistant');

      expect(openAIMessages).toEqual([
        { role: 'user', content: 'Hello' },
        { role: 'user', content: 'Hi there' },
        { role: 'user', content: 'How are you?' },
        { role: 'user', content: 'I am doing well, thank you!' },
      ]);
    });

    it('should handle empty messages array', () => {
      const emptyContext = new SessionContext('session-id', []);
      const openAIMessages = emptyContext.toOpenAI('assistant');

      expect(openAIMessages).toEqual([]);
    });

    it('should handle messages with missing peer_name', () => {
      const messagesWithMissingPeer = [
        { id: 'msg1', content: 'Hello', peer_name: 'assistant' },
        { id: 'msg2', content: 'No peer' }, // missing peer_name
        { id: 'msg3', content: 'Another message', peer_name: null },
      ];
      const context = new SessionContext('test', messagesWithMissingPeer);

      const openAIMessages = context.toOpenAI('assistant');

      expect(openAIMessages).toEqual([
        { role: 'assistant', content: 'Hello' },
        { role: 'user', content: 'No peer' },
        { role: 'user', content: 'Another message' },
      ]);
    });

    it('should handle complex message content', () => {
      const complexMessages = [
        { id: 'msg1', content: 'Message with\nnewlines and special chars!@#$%', peer_name: 'assistant' },
        { id: 'msg2', content: '', peer_name: 'user' }, // empty content
        { id: 'msg3', content: '   whitespace   ', peer_name: 'assistant' },
      ];
      const context = new SessionContext('test', complexMessages);

      const openAIMessages = context.toOpenAI('assistant');

      expect(openAIMessages).toEqual([
        { role: 'assistant', content: 'Message with\nnewlines and special chars!@#$%' },
        { role: 'user', content: '' },
        { role: 'assistant', content: '   whitespace   ' },
      ]);
    });
  });

  describe('toAnthropic', () => {
    it('should convert messages to Anthropic format with string assistant', () => {
      const anthropicMessages = sessionContext.toAnthropic('assistant');

      expect(anthropicMessages).toEqual([
        { role: 'assistant', content: 'Hello' },
        { role: 'user', content: 'Hi there' },
        { role: 'user', content: 'How are you?' },
        { role: 'assistant', content: 'I am doing well, thank you!' },
      ]);
    });

    it('should convert messages to Anthropic format with Peer object', () => {
      const mockHoncho = {} as any;
      const assistantPeer = new Peer('assistant', mockHoncho);

      const anthropicMessages = sessionContext.toAnthropic(assistantPeer);

      expect(anthropicMessages).toEqual([
        { role: 'assistant', content: 'Hello' },
        { role: 'user', content: 'Hi there' },
        { role: 'user', content: 'How are you?' },
        { role: 'assistant', content: 'I am doing well, thank you!' },
      ]);
    });

    it('should handle messages where assistant is different peer', () => {
      const anthropicMessages = sessionContext.toAnthropic('different-assistant');

      expect(anthropicMessages).toEqual([
        { role: 'user', content: 'Hello' },
        { role: 'user', content: 'Hi there' },
        { role: 'user', content: 'How are you?' },
        { role: 'user', content: 'I am doing well, thank you!' },
      ]);
    });

    it('should handle empty messages array', () => {
      const emptyContext = new SessionContext('session-id', []);
      const anthropicMessages = emptyContext.toAnthropic('assistant');

      expect(anthropicMessages).toEqual([]);
    });

    it('should handle messages with missing peer_name', () => {
      const messagesWithMissingPeer = [
        { id: 'msg1', content: 'Hello', peer_name: 'assistant' },
        { id: 'msg2', content: 'No peer' }, // missing peer_name
        { id: 'msg3', content: 'Another message', peer_name: undefined },
      ];
      const context = new SessionContext('test', messagesWithMissingPeer);

      const anthropicMessages = context.toAnthropic('assistant');

      expect(anthropicMessages).toEqual([
        { role: 'assistant', content: 'Hello' },
        { role: 'user', content: 'No peer' },
        { role: 'user', content: 'Another message' },
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
      expect(result).toBe('SessionContext(messages=4)');
    });

    it('should handle empty messages', () => {
      const emptyContext = new SessionContext('session-id', []);
      const result = emptyContext.toString();
      expect(result).toBe('SessionContext(messages=0)');
    });

    it('should handle large number of messages', () => {
      const manyMessages = Array.from({ length: 1000 }, (_, i) => ({
        id: `msg${i}`,
        content: `Message ${i}`,
        peer_name: i % 2 === 0 ? 'assistant' : 'user',
      }));
      const context = new SessionContext('session-id', manyMessages);

      const result = context.toString();
      expect(result).toBe('SessionContext(messages=1000)');
    });
  });

  describe('edge cases and error handling', () => {
    it('should handle messages with null content', () => {
      const messagesWithNullContent = [
        { id: 'msg1', content: null, peer_name: 'assistant' },
        { id: 'msg2', content: undefined, peer_name: 'user' },
      ];
      const context = new SessionContext('test', messagesWithNullContent);

      const openAIMessages = context.toOpenAI('assistant');

      expect(openAIMessages).toEqual([
        { role: 'assistant', content: null },
        { role: 'user', content: undefined },
      ]);
    });

    it('should handle messages with non-string content', () => {
      const messagesWithNonStringContent = [
        { id: 'msg1', content: 123, peer_name: 'assistant' },
        { id: 'msg2', content: { text: 'object content' }, peer_name: 'user' },
        { id: 'msg3', content: true, peer_name: 'assistant' },
      ];
      const context = new SessionContext('test', messagesWithNonStringContent);

      const openAIMessages = context.toOpenAI('assistant');

      expect(openAIMessages).toEqual([
        { role: 'assistant', content: 123 },
        { role: 'user', content: { text: 'object content' } },
        { role: 'assistant', content: true },
      ]);
    });

    it('should handle very long session IDs and summaries', () => {
      const longSessionId = 'x'.repeat(1000);
      const longSummary = 'Very long summary that goes on and on...'.repeat(100);
      const context = new SessionContext(longSessionId, mockMessages, longSummary);

      expect(context.sessionId).toBe(longSessionId);
      expect(context.summary).toBe(longSummary);
      expect(context.length).toBe(4);
    });

    it('should handle messages with additional properties', () => {
      const messagesWithExtraProps = [
        {
          id: 'msg1',
          content: 'Hello',
          peer_name: 'assistant',
          timestamp: '2023-01-01T00:00:00Z',
          metadata: { important: true },
          extra_field: 'extra_value'
        },
      ];
      const context = new SessionContext('test', messagesWithExtraProps);

      const openAIMessages = context.toOpenAI('assistant');

      expect(openAIMessages).toEqual([
        { role: 'assistant', content: 'Hello' },
      ]);
    });

    it('should handle case-sensitive peer names', () => {
      const caseMessages = [
        { id: 'msg1', content: 'Hello', peer_name: 'Assistant' },
        { id: 'msg2', content: 'Hi', peer_name: 'ASSISTANT' },
        { id: 'msg3', content: 'Hey', peer_name: 'assistant' },
      ];
      const context = new SessionContext('test', caseMessages);

      const openAIMessages = context.toOpenAI('assistant');

      expect(openAIMessages).toEqual([
        { role: 'user', content: 'Hello' }, // 'Assistant' != 'assistant'
        { role: 'user', content: 'Hi' }, // 'ASSISTANT' != 'assistant'
        { role: 'assistant', content: 'Hey' }, // exact match
      ]);
    });

    it('should handle messages without id field', () => {
      const messagesWithoutId = [
        { content: 'Message without ID', peer_name: 'assistant' },
        { peer_name: 'user', content: 'Another message' },
      ];
      const context = new SessionContext('test', messagesWithoutId);

      expect(context.length).toBe(2);
      expect(context.toOpenAI('assistant')).toEqual([
        { role: 'assistant', content: 'Message without ID' },
        { role: 'user', content: 'Another message' },
      ]);
    });
  });
});
