import os
import logging
import asyncio
from agents.mcp import MCPServerStdio

logger = logging.getLogger(__name__)
_mcp_server = None

async def initialize_mcp_server():
    """Initialize MCP server."""
    global _mcp_server
    
    if _mcp_server:
        return _mcp_server
    
    try:
        server = MCPServerStdio(
            cache_tools_list=False,
            params={
                # "command": "npx",
    #             "args": [
    #     "mcp-remote",
    #     "https://mcp.brightdata.com/mcp?token=72cdcb94a97dd019793ca6ffe5c8924ae59498b2e463c51b734fd37ec045e42b"
    #   ],
     "command": "npx",
      "args": ["@brightdata/mcp"],
      "env": {
        "API_TOKEN": "72cdcb94a97dd019793ca6ffe5c8924ae59498b2e463c51b734fd37ec045e42b",
        "WEB_UNLOCKER_ZONE": "mcp_unlocker",
        # "BROWSER_ZONE": "wss://brd-customer-hl_3ae08e24-zone-mcp_browser:v1iog5e9u55l@brd.superproxy.io:9222", 
        "PRO_MODE": "true"
      }
            }
        )
        
        await asyncio.wait_for(server.__aenter__(), timeout=10)
        _mcp_server = server
        return server
            
    except Exception as e:
        logger.error(f"Error initializing MCP server: {e}")
        return None

async def wait_for_initialization():
    """Wait for MCP initialization to complete."""
    return await initialize_mcp_server() is not None

def get_mcp_server():
    """Get the current MCP server instance."""
    return _mcp_server 