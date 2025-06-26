// Global test setup
import 'jest';

// Suppress console warnings during tests unless explicitly testing them
const originalConsoleWarn = console.warn;
const originalConsoleError = console.error;

beforeAll(() => {
  console.warn = jest.fn();
  console.error = jest.fn();
});

afterAll(() => {
  console.warn = originalConsoleWarn;
  console.error = originalConsoleError;
});

// Mock environment variables for tests
process.env.NODE_ENV = 'test'; 