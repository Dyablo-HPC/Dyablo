
/*
 * adapted from libb64 by CB
 */

/*
b64enc.c - c source to a base64 encoder

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
  char               *plaintext = 0;
  char               *code = 0;
  size_t              plainlength;
  size_t              codelength;
  base64_encodestate  state;

  code = (char *) malloc (sizeof (char) * readsize * 2);
  plaintext = (char *) malloc (sizeof (char) * readsize);

  base64_init_encodestate (&state);

  do {
    plainlength = fread ((void *) plaintext, sizeof (char), readsize, stdin);
    codelength = base64_encode_block (plaintext, plainlength, code, &state);
    (void) fwrite ((void *) code, sizeof (char), codelength, stdout);
  }
  while (!feof (stdin) && plainlength > 0);

  codelength = base64_encode_blockend (code, &state);
  (void) fwrite ((void *) code, sizeof (char), codelength, stdout);

  free (code);
  free (plaintext);

  return 0;
}
