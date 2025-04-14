import { SSTConfig } from "sst";
import { Api } from "sst/constructs";

export default {
  config(_input) {
    return {
      name: "doinstruct-cards",
      region: "us-east-1",
    };
  },
  stacks(app) {
    app.stack(function Stack({ stack }) {
      const api = new Api(stack, "api", {
        defaults: {
          function: {
            runtime: "nodejs18.x",
            environment: {
              OPENAI_API_KEY: process.env.OPENAI_API_KEY || "",
              NODE_ENV: "production"
            },
            copyFiles: [
              {
                from: "pdf_inputs",
                to: "pdf_inputs"
              }
            ]
          },
        },
        cors: {
          allowMethods: ["POST"],
          allowOrigins: ["*"],
        },
        routes: {
          "POST /generate-cards": "src/lambda.handler",
        },
      });

      stack.addOutputs({
        ApiEndpoint: api.url,
      });
    });
  },
} satisfies SSTConfig;