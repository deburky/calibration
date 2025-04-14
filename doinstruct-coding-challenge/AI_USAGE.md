# AI Tools Usage

This project was developed with the assistance of AI tools. Below is a detailed breakdown of how AI was used:

## Development Assistance

1. **Code Generation**
   - Initial project structure and setup
   - TypeScript type definitions
   - API endpoint implementation
   - Error handling patterns

2. **Code Review and Optimization**
   - Suggestions for code organization
   - Best practices implementation
   - Error handling improvements

3. **Documentation**
   - README.md content generation
   - API documentation
   - Code comments

## Specific AI Tools Used

1. **Cursor AI Assistant**
   - Used for code generation and review
   - Assisted with TypeScript type definitions
   - Helped with AWS Lambda and SST configuration
   - Provided suggestions for error handling and validation

2. **OpenAI GPT-4**
   - Used for generating learning card content
   - Implemented in the `llmService.ts` file
   - Handles the actual card generation based on user input

## Prompt Engineering

The system uses carefully crafted prompts to generate high-quality learning cards:

1. **System Prompt**
   - Defines the role of the AI as an educational content creator
   - Sets guidelines for card creation
   - Ensures content is suitable for professional learning environments

2. **User Prompt**
   - Combines user input with context
   - Provides clear instructions for card generation
   - Includes language-specific requirements

### Model Selection & Iterations

I chose OpenAI's GPT-4 over Mistral for this implementation due to:
- Superior JSON structure reliability and consistency
- Better handling of multilingual content (especially German safety terminology)
- More consistent adherence to length constraints for titles and content

Prompt iteration process:
1. First version lacked clear content length constraints → Added explicit character limits
2. Initial outputs were too general → Added safety-specific terminology requirements for each language
3. Early responses sometimes lacked structured JSON → Added explicit JSON format instructions
4. Content sometimes lacked specific safety focus → Enhanced prompt with domain-specific guidance

## Ethical Considerations

- All AI-generated content is reviewed and validated
- The system includes proper error handling for AI responses
- User data is handled securely and in compliance with privacy standards
- The system is designed to be transparent about AI usage 