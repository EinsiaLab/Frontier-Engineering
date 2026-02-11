#!/bin/bash

# Prompt for API KEY
while true; do
    read -p "Please enter your API KEY: " API_KEY
    if [ -n "$API_KEY" ]; then
        break
    else
        echo "Error: API KEY cannot be empty. Please try again."
    fi
done

# Check if config files exist and backup if necessary
CONFIG_FILE=~/.codex/config.toml
AUTH_FILE=~/.codex/auth.json
BACKUP_FILE=~/codex-config.backup.tar.gz

if [ -f "$CONFIG_FILE" ] || [ -f "$AUTH_FILE" ]; then
    echo "Existing configuration files found. Creating backup..."
    tar -czf "$BACKUP_FILE" -C ~/.codex $(ls ~/.codex/ | grep -E '^(config\.toml|auth\.json)$' 2>/dev/null)
    echo "Backup created: $BACKUP_FILE"
fi

# Create ~/.codex directory if it doesn't exist
mkdir -p ~/.codex

# Create config.toml
cat > "$CONFIG_FILE" << 'EOF'
model_provider = "codex-for-me"
model = "gpt-5.3-codex"
model_reasoning_effort = "xhigh"
disable_response_storage = true

[model_providers.codex-for-me]
name = "codex-for-me"
base_url = "https://api-vip.codex-for.me/v1"
wire_api = "responses"
requires_openai_auth = true
EOF

# Create auth.json with user-provided API KEY
cat > "$AUTH_FILE" << EOF
{
"OPENAI_API_KEY": "$API_KEY"
}
EOF

echo ""
echo "Configuration completed successfully!"
echo ""
echo "To restore previous configuration, run:"
echo "  tar -xzf ~/codex-config.backup.tar.gz -C ~/.codex"
echo ""
