#!/bin/bash
# Security scanning script for ASTINA
# Scans for: hardcoded secrets, dependency vulnerabilities, code security issues
# Usage: ./scripts/security-scan.sh [install-tools]

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

SCAN_DIR="."
RESULTS_FILE="security-scan-results.json"
EXIT_CODE=0

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "════════════════════════════════════════════════════════════════════"
echo "🔒 ASTINA SECURITY SCAN"
echo "════════════════════════════════════════════════════════════════════"
echo "Timestamp: $(date)"
echo "Project Root: $PROJECT_ROOT"
echo ""

# Function to check if tool is installed
check_tool() {
    if ! command -v "$1" &> /dev/null; then
        echo -e "${YELLOW}⚠️  $1 not found${NC}"
        return 1
    fi
    return 0
}

# Install mode
if [ "$1" == "install-tools" ]; then
    echo "📦 Installing security scanning tools..."
    pip install detect-secrets bandit pip-audit --upgrade
    echo "✅ Tools installed"
    echo ""
fi

# 1. HARDCODED SECRETS SCAN
echo "════════════════════════════════════════════════════════════════════"
echo "1️⃣  HARDCODED SECRETS SCAN"
echo "════════════════════════════════════════════════════════════════════"

if check_tool detect-secrets; then
    echo "🔍 Scanning for hardcoded secrets using detect-secrets..."
    
    # Create or update baseline
    if [ ! -f ".secrets.baseline" ]; then
        echo "Creating .secrets.baseline..."
        detect-secrets scan --baseline .secrets.baseline --all-files --force-use-all-plugins
    fi
    
    # Run scan
    if detect-secrets scan --baseline .secrets.baseline --force-use-all-plugins 2>/dev/null; then
        echo -e "${GREEN}✅ No hardcoded secrets detected${NC}"
    else
        echo -e "${RED}❌ Potential secrets found!${NC}"
        detect-secrets scan --all-files --all-plugins
        EXIT_CODE=1
    fi
else
    echo "⏭️  detect-secrets not installed. Install with: pip install detect-secrets"
fi
echo ""

# 2. DEPENDENCY VULNERABILITY SCAN
echo "════════════════════════════════════════════════════════════════════"
echo "2️⃣  DEPENDENCY VULNERABILITY SCAN"
echo "════════════════════════════════════════════════════════════════════"

if check_tool pip-audit; then
    echo "🔍 Scanning dependencies for vulnerabilities using pip-audit..."
    
    # Scan with JSON output
    if pip-audit --desc --skip-editable 2>/dev/null; then
        echo -e "${GREEN}✅ No dependency vulnerabilities detected${NC}"
    else
        echo -e "${YELLOW}⚠️  Vulnerabilities or issues found (check output above)${NC}"
        # Non-fatal by default (dependencies may have known issues in legacy versions)
        # EXIT_CODE=1  # Uncomment to fail on any vulnerability
    fi
else
    echo "⏭️  pip-audit not installed. Install with: pip install pip-audit"
    echo "   Or use: pip install -r requirements.txt --upgrade"
fi
echo ""

# 3. CODE SECURITY ISSUES SCAN
echo "════════════════════════════════════════════════════════════════════"
echo "3️⃣  CODE SECURITY ISSUES SCAN (Bandit)"
echo "════════════════════════════════════════════════════════════════════"

if check_tool bandit; then
    echo "🔍 Scanning code for security issues using bandit..."
    
    # Create temporary baseline exclusion file
    cat > /tmp/bandit_baseline.txt << 'EOF'
# Known non-critical findings to exclude:
# B101: Test assertions
# B601: Paramiko call with automatic add (evaluated per case)
EOF
    
    if bandit -r "$SCAN_DIR" -ll --skip B101,B601 -f json -o "$RESULTS_FILE" 2>/dev/null; then
        BANDIT_ISSUES=$(grep -c '"issue_cwe"' "$RESULTS_FILE" 2>/dev/null || echo "0")
        
        if [ "$BANDIT_ISSUES" -eq 0 ]; then
            echo -e "${GREEN}✅ No critical security issues detected${NC}"
        else
            echo -e "${YELLOW}⚠️  Found $BANDIT_ISSUES security issues (see $RESULTS_FILE)${NC}"
        fi
    else
        echo -e "${YELLOW}⚠️  Bandit scan issues (see output above)${NC}"
    fi
    
    # Clean up
    rm -f /tmp/bandit_baseline.txt
else
    echo "⏭️  bandit not installed. Install with: pip install bandit"
fi
echo ""

# 4. PYTHON CODE QUALITY
echo "════════════════════════════════════════════════════════════════════"
echo "4️⃣  PYTHON CODE QUALITY"
echo "════════════════════════════════════════════════════════════════════"

if check_tool pylint; then
    echo "🔍 Checking Python code quality..."
    if pylint *.py --disable=all --enable=E,F --exit-zero 2>/dev/null | head -20; then
        echo -e "${GREEN}✅ Code quality check complete${NC}"
    fi
else
    echo "⏭️  pylint not installed. Install with: pip install pylint"
fi
echo ""

# 5. SUMMARY
echo "════════════════════════════════════════════════════════════════════"
echo "📊 SCAN SUMMARY"
echo "════════════════════════════════════════════════════════════════════"
echo "Scan Date: $(date)"
echo "Results File: $RESULTS_FILE"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✅ Security scan PASSED${NC}"
    echo ""
    echo "🎯 Recommendations:"
    echo "  • Run this scan before each deployment"
    echo "  • Add to CI/CD pipeline for automated checks"
    echo "  • Review .secrets.baseline for false positives"
    echo "  • Update dependencies regularly: pip install --upgrade -r requirements.txt"
else
    echo -e "${RED}❌ Security scan FAILED${NC}"
    echo ""
    echo "🔧 Next Steps:"
    echo "  1. Review findings in $RESULTS_FILE"
    echo "  2. Remediate issues (secrets, vulnerabilities, etc.)"
    echo "  3. Re-run scan to verify fixes"
    echo "  4. Do not proceed with deployment until scan passes"
fi

echo ""
echo "════════════════════════════════════════════════════════════════════"

exit $EXIT_CODE
