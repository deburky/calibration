import { StackContext, Api } from "sst/constructs";

export function API({ stack }: StackContext) {
  const api = new Api(stack, "api", {
    defaults: {
      function: {
        runtime: "nodejs18.x",
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
}