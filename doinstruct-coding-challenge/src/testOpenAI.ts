import OpenAI from 'openai';
import dotenv from 'dotenv';

dotenv.config();

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

async function testOpenAI() {
  try {
    const response = await openai.chat.completions.create({
      model: 'gpt-4',
      messages: [
        {
          role: 'user',
          content: 'Hello, this is a test message. Please respond with "API key is working!"'
        }
      ],
      temperature: 0.7,
    });

    console.log('OpenAI API Response:', response.choices[0].message.content);
    console.log('API key is valid and working!');
  } catch (error) {
    console.error('Error testing OpenAI API:', error);
  }
}

testOpenAI(); 