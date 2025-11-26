import traceback
import inspect
from colorama import Fore, Style
from textwrap import indent
from colorama import init
init(autoreset=True)  # Initialize Colorama (auto-reset ensures colors don't '


def handle_error(exc, def_name=None, msg='', show_raw=False):
    exc_type = type(exc)
    exc_value = exc
    exc_tb = exc.__traceback__

    tb_str = ''.join(traceback.format_exception(exc_type, exc_value, exc_tb))

    if def_name is None:
        def_name = inspect.stack()[1].function

    print('\n' + Fore.RED + '=' * 100)
    print(Fore.RED + Style.BRIGHT + f'🚨  ERROR OCCURRED IN {def_name}()')

    if msg:
        print(Fore.YELLOW + f'Context: {msg}')

    print(Fore.RED + '-' * 100)
    print(Fore.CYAN + f'Type: {exc_type.__name__}')
    print(Fore.MAGENTA + f'Message: {exc_value}')
    print(Fore.RED + '-' * 100)

    print(Fore.WHITE + Style.BRIGHT + 'Traceback (most recent call last):\n')
    print(indent(Fore.LIGHTBLACK_EX + tb_str.strip(), '  '))
    print(Fore.RED + '=' * 100 + Style.RESET_ALL + '\n')

    if show_raw:
        print('Full Raw Traceback:')
        print(tb_str)