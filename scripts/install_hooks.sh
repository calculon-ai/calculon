#!/bin/bash

set -e

# Pre-commit hook
cat > .git/hooks/pre-commit <<-EOF
#!/bin/bash
echo -n "Testing..."
if ! make test &> /dev/null; then
    echo " failed :("
    exit -1
fi
EOF

chmod a+x .git/hooks/pre-commit
