FROM mcr.microsoft.com/devcontainers/base:debian

ARG USERNAME=vscode

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get upgrade -y 


RUN SNIPPET="export PROMPT_COMMAND='history -a' && export HISTFILE=/commandhistory/.bash_history" \
        && mkdir /commandhistory \
        && touch /commandhistory/.bash_history \
        && chown -R $USERNAME /commandhistory \
        && echo "$SNIPPET" >> "/home/$USERNAME/.bashrc" \
        && echo 'bind "\"\e[A\": history-search-backward"' >> "/home/$USERNAME/.bashrc" \
        && echo 'bind "\"\e[B\": history-search-forward"' >> "/home/$USERNAME/.bashrc"