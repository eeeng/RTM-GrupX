from uagents import Agent, Context

alice = Agent(name= "alice", seed= "alice recovery phrase")

@alice.on_interval(period=2)

async def say_hello(ctx):
    print(dir(ctx))  
    if hasattr(ctx, 'agent_name'):
        ctx.logger.info(f'Hello, my name is {ctx.agent_name}')
    else:
        ctx.logger.info('Hello, my name is ahmet sari')

if __name__ == "__main__":
    alice.run()