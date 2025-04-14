# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build/Test/Lint Commands

- Start local server: `npm start` (required for tests to run)
- Run tests: `npm test` or `npm run test` (requires local server to be running)
- Run tests in watch mode: `npm run test:watch` (requires local server to be running)
- Run a specific test file: `npx jest src/tests/path-to-file.test.ts` (requires local server to be running)
- Run a specific test: `npx jest -t "test description"` (requires local server to be running)
- TypeScript type checking: `npx tsc --noEmit`

## Code Style Guidelines

- TypeScript version: ES2020 target with strict type checking
- Line length: Follow existing file conventions (typically 80-100 chars)
- Indentation: 2 spaces
- Imports: External packages first, then local modules using relative paths
- Naming: camelCase for variables/functions, PascalCase for types/interfaces/classes
- Constants: Use ALL_CAPS for unchanging values
- Error handling: Use explicit try/catch with descriptive messages and appropriate HTTP status codes
- Types: Always use type annotations for function parameters and return values
- Testing: Use Jest with describe/it blocks and expect() assertions
- Async: Use async/await for asynchronous operations
- Code organization: Modular structure with separation of concerns (types, services, functions)

## Project Architecture

This project implements a serverless API for generating educational learning cards using LLMs:

- **src/functions/** - Contains AWS Lambda handler functions
- **src/services/** - Contains service modules (LLM integration, PDF processing)
- **src/types/** - Contains TypeScript type definitions
- **src/tests/** - Contains Jest test files
- **sst-deployment/** - Contains a standalone mock implementation that doesn't require API keys

## LLM Prompt Engineering Guidelines

When modifying LLM-related code, follow these guidelines:

1. **System and User Prompts**:
   - Maintain separation between system and user prompts
   - System prompts define the role and output format
   - User prompts combine dynamic content with specific instructions

2. **Output Format Specification**:
   - Always include explicit JSON structure requirements in system prompts
   - Ensure content and title length constraints are mentioned (100 char title, 500 char content)
   - Validate and clean LLM responses before returning to clients

3. **Language-Specific Considerations**:
   - Each supported language (en, de, es, fr) should include tailored terminology
   - German safety terms require special attention for technical accuracy
   - Error messages should match the user's requested language when possible

4. **Error Handling**:
   - Implement proper try/catch blocks around LLM API calls
   - Validate LLM responses against expected schema
   - Have fallback mechanisms for malformed responses

## Deployment Strategy

The project uses SST (Serverless Stack) for AWS deployment:

- **Development**: Use `npm run dev` in the sst-deployment directory
- **Building**: Use `npm run build` to prepare for deployment without AWS charges
- **Production**: Use `npm run deploy` to deploy to AWS

The mock implementation in sst-deployment provides a simulated version that doesn't require OpenAI API keys, ideal for testing the deployment process without incurring charges.