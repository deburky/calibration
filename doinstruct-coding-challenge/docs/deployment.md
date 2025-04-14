# Deployment Documentation

## Local Development

The application can be run locally using:
```bash
npm start
```
This starts the server at `http://localhost:3000`

## SST Deployment

The application has been successfully deployed using SST (Serverless Stack) in development mode.

### Deployment Details
- **App Name**: doinstruct-cards
- **Stage**: dev
- **Region**: us-east-1
- **API Endpoint**: https://vawtgqcoj5.execute-api.us-east-1.amazonaws.com
- **SST Console**: https://console.sst.dev/local/doinstruct-cards/dev

### Deployed Resources
- API Gateway endpoint
- Lambda function for card generation
- IAM roles and policies
- CloudWatch log group

### Cost Management
The deployment was done in development mode (`npm run dev -- --stage dev`), which:
- Uses local resources for development
- Doesn't create permanent AWS resources
- Runs in a sandboxed environment
- Minimizes AWS charges

### Cleanup
To clean up the development environment:
```bash
npm run remove
```

## Testing
Both local and SST deployments have been tested and verified:
- Local tests: 9/9 tests passing
- SST deployment: Successfully deployed and accessible
- API endpoints working in both environments
- Card generation functioning correctly
- Safety terminology properly enforced
- PDF generation working

## Notes
- The SST deployment is in development mode only
- No production resources have been created
- AWS charges are minimized by using development mode
- Local development is recommended for most testing 