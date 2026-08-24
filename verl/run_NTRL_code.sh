#! /usr/bin/env bash
IFS=$'\n\t'

# color
BOLD='\033[1m'
GREEN='\033[32m'
YELLOW='\033[33m'
RED='\033[31m'
CYAN='\033[36m'
RESET='\033[0m'

separator(){
    local char="${1:-=}"
    local count="${2:-60}"
    printf '%*s\n' "${count}" '' | tr ' ' "${char}"
}

time_stamp(){
    date +"%Y-%m-%d %H:%M:%S"
}

announce(){
    local msg="$1"
    separator "=" 60
    printf "${BOLD}${CYAN}[%s] %s${RESET}\n" "$(time_stamp)" "${msg}"
    separator "=" 60
}

run_step(){
    local label="$1"
    local script_path="$2"

    announce "Run NTRL-Code for ${label}"
    echo -e "${YELLOW}Execting:${RESET} bash ${script_path}"
    if [[ ! -f "${script_path}" ]]; then
        echo -e "${RED}Error:${RESET} Script not found: ${script_path}"
        return 1
    fi

    bash "${script_path}"
    local rc=$?

    if [[ ${rc} -eq 0 ]]; then
        echo -e "${GREEN}Success:${RESET} ${label}"
    else
        echo -e "${RED}Failed (${rc}):${RESET} ${label}"
        read -r -p "Continue with next step? [Y/n] " ans
        ans=${ans:-Y}
        if [[ ! "${ans}" =~ ^[Yy]$ ]]; then
            echo "Aborting."
            exit $rc
        fi
    fi
    
    echo 
}


run_step "NTRL-Code HumanEval" "examples/ntrl_code/Deepseek-Coder-1.3B/humaneval_ntrl_code.sh"

echo -e "${BOLD}${GREEN}All steps completed.${RESET}"