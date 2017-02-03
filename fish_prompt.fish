function fish_prompt
      if not set -q __fish_prompt_hostname
                    set -g __fish_prompt_hostname (hostname | cut -d"." -f1)
                   end
                        set_color -o green
                             echo -n -s "$USER"

                             echo -n " "
                             set_color -o yellow
                             echo -n "$__fish_prompt_hostname" ""
                                  set_color -o blue
                                       echo -n (prompt_pwd)
                                            echo -n " % "
                                                 set_color normal
                                                 end