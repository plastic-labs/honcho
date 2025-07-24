# Honcho TypeScript SDK Test Suite

This directory contains an exhaustive and idiomatic test suite for the Honcho TypeScript SDK that covers every available endpoint in multiple ways.

## Test Structure

### Unit Tests

Each class has its own dedicated test file with comprehensive coverage:

- **`client.test.ts`** - Tests for the main `Honcho` client class
- **`peer.test.ts`** - Tests for the `Peer` class
- **`session.test.ts`** - Tests for the `Session` class
- **`session_context.test.ts`** - Tests for the `SessionContext` class
- **`pagination.test.ts`** - Tests for the `Page` class

### Integration Tests

- **`integration.test.ts`** - End-to-end workflow tests that demonstrate real-world usage patterns

### Test Configuration

- **`setup.ts`** - Global test setup and configuration
- **`jest.config.js`** - Jest configuration
- **`__mocks__/@honcho-ai/core.ts`** - Mock implementation of the core API client

## Test Coverage

### Honcho Client (`client.test.ts`)

- ✅ Constructor with all option variations
- ✅ Environment variable fallbacks
- ✅ Peer creation and validation
- ✅ Session creation and validation
- ✅ Workspace metadata operations
- ✅ Workspace listing
- ✅ Search functionality
- ✅ Error handling for all methods
- ✅ Edge cases and input validation

### Peer Class (`peer.test.ts`)

- ✅ Chat functionality with all option combinations
- ✅ Session management
- ✅ Message operations (add, get, create)
- ✅ Metadata operations
- ✅ Search within peer scope
- ✅ Different input types and formats
- ✅ Null/empty response handling
- ✅ Error scenarios

### Session Class (`session.test.ts`)

- ✅ Peer management (add, set, remove, list)
- ✅ Message operations with filtering
- ✅ Metadata operations
- ✅ Context retrieval with options
- ✅ Search within session scope
- ✅ Working representation queries
- ✅ Mixed input types (strings vs objects)
- ✅ Constructor options
- ✅ Error handling

### SessionContext Class (`session_context.test.ts`)

- ✅ Constructor variations
- ✅ OpenAI format conversion
- ✅ Anthropic format conversion
- ✅ Length and toString methods
- ✅ Empty/null message handling
- ✅ Complex message content
- ✅ Case sensitivity
- ✅ Missing field handling
- ✅ Edge cases and malformed data

### Page Class (`pagination.test.ts`)

- ✅ Async iteration
- ✅ Transform functions
- ✅ Data retrieval methods
- ✅ Pagination navigation
- ✅ Different page formats
- ✅ Error handling in transforms
- ✅ Large dataset handling
- ✅ Circular references
- ✅ Complex nested structures

### Integration Tests (`integration.test.ts`)

- ✅ Complete chat session workflow
- ✅ Workspace and peer management
- ✅ Multi-scope search functionality
- ✅ Error scenario handling
- ✅ Pagination workflows
- ✅ Working representation queries
- ✅ Type safety verification
- ✅ Empty/null response handling

## Test Patterns

### Comprehensive Mocking

All tests use comprehensive mocks of the underlying `@honcho-ai/core` API client to ensure:

- Tests run independently of external services
- Predictable and controllable test scenarios
- Fast test execution
- Ability to test error conditions

### Edge Case Coverage

Each test suite includes extensive edge case testing:

- Empty/null inputs and responses
- Invalid input types
- API error conditions
- Boundary conditions
- Malformed data handling

### Multiple Input Types

Tests verify that methods handle various input types correctly:

- String vs object parameters
- Single items vs arrays
- Optional vs required parameters
- Different data structures

### Error Scenarios

Comprehensive error testing including:

- API failures
- Invalid inputs
- Network errors
- Timeout scenarios
- Validation errors

### Async Operations

Proper testing of all asynchronous operations:

- Promise resolution/rejection
- Async iteration
- Concurrent operations
- Error propagation

## Running Tests

```bash
# Install dependencies
npm install

# Run all tests
npm test

# Run tests with coverage
npm run test:coverage

# Run tests in watch mode
npm run test:watch

# Run specific test file
npm test client.test.ts

# Run integration tests only
npm test integration.test.ts
```

## Coverage Goals

This test suite aims for:

- **100% function coverage** - Every function is called
- **100% branch coverage** - Every code path is tested
- **100% statement coverage** - Every line is executed
- **Comprehensive edge case coverage** - Every failure mode is tested

## Test Philosophy

1. **Exhaustive Testing**: Every public method and property is tested
2. **Multiple Scenarios**: Each method is tested with various inputs and conditions
3. **Real-world Usage**: Integration tests mirror actual usage patterns
4. **Error Resilience**: Extensive error condition testing
5. **Type Safety**: TypeScript types are verified throughout
6. **Performance Awareness**: Tests include large dataset scenarios
7. **Maintainability**: Clear, well-documented test cases

## Mock Strategy

The test suite uses a sophisticated mocking strategy:

1. **Core API Mocking**: The `@honcho-ai/core` module is completely mocked
2. **Flexible Responses**: Mock responses can be configured per test
3. **Error Simulation**: Easy simulation of API errors and edge cases
4. **Isolation**: Each test runs in complete isolation
5. **Deterministic**: Tests produce consistent, reproducible results

This ensures that the SDK layer is thoroughly tested while remaining independent of the underlying API implementation.
