<h1 align="center">
  SUSTAINER - Modulo FIA<br>
</h1>

<img src="https://github.com/xrenegade100/sustainer/assets/11615441/452ab4ec-b4b3-4858-ace3-e4f0195d9b34">

# Indice

-   [Prerequisiti](#prerequisiti)
-   [Installazione](#installazione)
-   [Come contribuire](#come-contribuire)
-   [Team](#team)

# Prerequisiti

-   Python 3.11.3
-   R 4.3.2

**Nota bene**: Durante l'installazione del package `rpy2` 
ti saranno richieste le seguenti librerie: `libpcre2-dev`, `libbz2-dev` e `zlib1g-dev`.
Verifca dunque la loro presenza nel seguente modo.


### Windows:
1. Imposta la variabile d'ambiente `R_HOME`
2. Aggiungi `R_HOME\bin` a `PATH` in modo da eseguire `R` da `Python`
3. Aggiungi `R_HOME\bin\x64` a `PATH` in modo da caricare `R.dll`
4. Riavvia il PC

### MacOS:
1. Installa Homebrew da terminale:
```bash
 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```
2. Installa le librerie richieste
```bash
 brew install pcre2 bzip2 zlib
```




# Installazione

1. Clona questa repository

```bash
 git clone https://github.com/MegaPepeMan/sustainer_moduloFIA.git
```

2. Installa le dipendenze per il backend

```bash
 pip3 install -r requirements.txt
```

3. Esegui il modulo per il test

```bash
 cd tests
 python3 test-1.py
```

# Come contribuire

**Importante: leggi [CONTRUTING](docs/CONTRIBUTING.md)**

# Team

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->

<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%">
        <a href="https://github.com/MegaPepeMan"><img src="https://avatars.githubusercontent.com/u/83645460?v=4" width="100px;"/><br /><sub><b>Giuseppe Raiola Paduano</b></sub></a>
      </td>
      <td align="center" valign="top" width="14.28%">
        <a href="https://github.com/domenicod25"><img src="https://avatars.githubusercontent.com/u/137888029?v=4" width="100px;"/><br /><sub><b>Domenico D'Urso</b></sub></a>
      </td>
      <td align="center" valign="top" width="14.28%">
        <a href="https://github.com/RaffyAS99"><img src="https://avatars.githubusercontent.com/u/114479230?v=4" width="100px;"/><br /><sub><b>Raffaele Vietri</b></sub></a>
      </td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->
