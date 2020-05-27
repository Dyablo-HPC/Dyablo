
/*
 * adapted from libb64 by CB
 */

/*
b64dec.c - c source to a base64 decoder

This is part of the libb64 project, and has been placed in the public domain.
For details, see http://sourceforge.net/projects/libb64
*/

#include <libb64.h>
#include <stdio.h>
#include <stdlib.h>

int
main (void)
{
  const size_t        readsize = 4096;
  char               *code = 0;
  char               *plaintext = 0;
  size_t              codelength;
  size_t              plainlength;
  base64_decodestate  state;

  code = (char *) malloc (sizeof (char) * readsize);
  plaintext = (char *) malloc (sizeof (char) * readsize);

  base64_init_decodestate (&state);

  do {
    codelength = fread ((void *) code, sizeof (char), readsize, stdin);
    plainlength = base64_decode_block (code, codelength, plaintext, &state);
    (void) fwrite ((void *) plaintext, sizeof (char), plainlength, stdout);
  }
  while (!feof (stdin) && codelength > 0);

  free (code);
  free (plaintext);

  return 0;
}
