/*
 * MIT License
 *
 * Copyright (c) 2010 Serge Zaitsev
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "jsmn.h"

/**
 * Allocates a fresh unused token from the token pool.
 */
static jsmntok_t *jsmn_alloc_token(jsmn_parser *parser, jsmntok_t *tokens,
		size_t num_tokens) {
	jsmntok_t *tok;
	if (parser->toknext >= num_tokens) {
		return NULL;
	}
	tok = &tokens[parser->toknext++];
	tok->start = tok->end = -1;
	tok->size = 0;
#ifdef JSMN_PARENT_LINKS
	tok->parent = -1;
#endif
	return tok;
}

/**
 * Fills next available token with JSON primitive.
 */
static int jsmn_parse_primitive(jsmn_parser *parser, const char *js,
		size_t len, jsmntok_t *tokens, size_t num_tokens) {
	jsmntok_t *tok;
	int start;

	start = parser->pos;

	for (; parser->pos < len && js[parser->pos] != '\0'; parser->pos++) {
		switch (js[parser->pos]) {
#ifndef JSMN_STRICT
		/* In non-strict mode every unquoted value is a primitive */
		case '\t':
		case '\r':
		case '\n':
		case ' ':
		case ',':
		case ']':
		case '}':
#endif
		case ':':
			goto found;
		}
		if (js[parser->pos] < 32 || js[parser->pos] >= 127) {
			parser->pos = start;
			return JSMN_ERROR_INVAL;
		}
	}
#ifdef JSMN_STRICT
	/* In strict mode primitive must be followed by a comma/object/array */
	parser->pos = start;
	return JSMN_ERROR_PART;
#else
found:
#endif
	if (tokens == NULL) {
		parser->pos--;
		return 0;
	}
	tok = jsmn_alloc_token(parser, tokens, num_tokens);
	if (tok == NULL) {
		parser->pos = start;
		return JSMN_ERROR_NOMEM;
	}
	tok->type = JSMN_PRIMITIVE;
	tok->start = start;
	tok->end = parser->pos;
	tok->size = 0;
#ifdef JSMN_PARENT_LINKS
	tok->parent = parser->toksuper;
#endif
	parser->pos--;
	return 0;
}

/**
 * Fills next token with JSON string.
 */
static int jsmn_parse_string(jsmn_parser *parser, const char *js,
		size_t len, jsmntok_t *tokens, size_t num_tokens) {
	jsmntok_t *tok;

	int start = parser->pos;
	parser->pos++;

	/* Skip starting quote */
	for (; parser->pos < len && js[parser->pos] != '\0'; parser->pos++) {
		char c = js[parser->pos];

		/* Quote: end of string */
		if (c == '"') {
			if (tokens == NULL) {
				return 0;
			}
			tok = jsmn_alloc_token(parser, tokens, num_tokens);
			if (tok == NULL) {
				parser->pos = start;
				return JSMN_ERROR_NOMEM;
			}
			tok->type = JSMN_STRING;
			tok->start = start + 1;
			tok->end = parser->pos;
			tok->size = 0;
#ifdef JSMN_PARENT_LINKS
			tok->parent = parser->toksuper;
#endif
			return 0;
		}

		/* Backslash: Quoted symbol expected */
		if (c == '\\' && parser->pos + 1 < len) {
			int i;
			parser->pos++;
			switch (js[parser->pos]) {
			case '\"': case '/': case '\\': case 'b':
			case 'f': case 'r': case 'n': case 't':
				break;
			case 'u':
				parser->pos++;
				for(i = 0; i < 4 && parser->pos < len && js[parser->pos] != '\0'; i++) {
					if(!((js[parser->pos] >= '0' && js[parser->pos] <= '9') ||
					      (js[parser->pos] >= 'a' && js[parser->pos] <= 'f') ||
					      (js[parser->pos] >= 'A' && js[parser->pos] <= 'F'))) {
						parser->pos = start;
						return JSMN_ERROR_INVAL;
					}
					parser->pos++;
				}
				parser->pos--;
				break;
			default:
				parser->pos = start;
				return JSMN_ERROR_INVAL;
			}
		}
	}
	parser->pos = start;
	return JSMN_ERROR_PART;
}

/**
 * Parse JSON string and fill tokens.
 */
int jsmn_parse(jsmn_parser *parser, const char *js, size_t len,
		jsmntok_t *tokens, unsigned int num_tokens) {
	int r;
	int i;
	jsmntok_t *tok;
	size_t count = parser->toknext;

	for (; parser->pos < len && js[parser->pos] != '\0'; parser->pos++) {
		char c;
		jsmntype_t type;

		c = js[parser->pos];
		switch (c) {
		case '{': case '[':
			count++;
			if (tokens == NULL) {
				break;
			}
			tok = jsmn_alloc_token(parser, tokens, num_tokens);
			if (tok == NULL) {
				return JSMN_ERROR_NOMEM;
			}
			if (parser->toksuper != -1) {
				jsmntok_t *t = &tokens[parser->toksuper];
#ifdef JSMN_STRICT
				/* In strict mode an object or array can't become a key */
				if (t->type == JSMN_OBJECT) {
					return JSMN_ERROR_INVAL;
				}
#endif
				t->size++;
#ifdef JSMN_PARENT_LINKS
				tok->parent = parser->toksuper;
#endif
			}
			tok->type = (c == '{' ? JSMN_OBJECT : JSMN_ARRAY);
			tok->start = parser->pos;
			parser->toksuper = parser->toknext - 1;
			break;
		case '}': case ']':
			if (tokens == NULL) {
				break;
			}
			type = (c == '}' ? JSMN_OBJECT : JSMN_ARRAY);
#ifdef JSMN_PARENT_LINKS
			if (parser->toknext < 1) {
				return JSMN_ERROR_INVAL;
			}
			tok = &tokens[parser->toknext - 1];
			for (;;) {
				if (tok->start != -1 && tok->end == -1) {
					if (tok->type != type) {
						return JSMN_ERROR_INVAL;
					}
					tok->end = parser->pos + 1;
					if (parser->toksuper != -1) {
						parser->toksuper = tok->parent;
					} else {
						break;
					}
					tok = &tokens[parser->toksuper];
				} else {
					jsmntok_t *parent = &tokens[parser->toksuper];
					if (parent->start != -1 && parent->end == -1) {
						if (parent->type != type) {
							return JSMN_ERROR_INVAL;
						}
						parent->end = parser->pos + 1;
						if (parser->toksuper != -1) {
							parser->toksuper = parent->parent;
						}
					}
					break;
				}
			}
#else
			for (i = parser->toknext - 1; i >= 0; i--) {
				tok = &tokens[i];
				if (tok->start != -1 && tok->end == -1) {
					if (tok->type != type) {
						return JSMN_ERROR_INVAL;
					}
					parser->toksuper = -1;
					tok->end = parser->pos + 1;
					break;
				}
			}
			/* Error if unmatched closing bracket */
			if (i == -1) {
				return JSMN_ERROR_INVAL;
			}
			for (; i >= 0; i--) {
				tok = &tokens[i];
				if (tok->start != -1 && tok->end == -1) {
					parser->toksuper = i;
					break;
				}
			}
#endif
			break;
		case '\"':
			r = jsmn_parse_string(parser, js, len, tokens, num_tokens);
			if (r < 0) {
				return r;
			}
			count++;
			if (parser->toksuper != -1 && tokens != NULL) {
				tokens[parser->toksuper].size++;
			}
			break;
		case '\t': case '\r': case '\n': case ' ':
			break;
		case ':':
			parser->toksuper = parser->toknext - 1;
			break;
		case ',':
			if (tokens != NULL && parser->toksuper != -1 &&
					tokens[parser->toksuper].type != JSMN_ARRAY &&
					tokens[parser->toksuper].type != JSMN_OBJECT) {
#ifdef JSMN_PARENT_LINKS
				parser->toksuper = tokens[parser->toksuper].parent;
#else
				for (i = parser->toknext - 1; i >= 0; i--) {
					if (tokens[i].type == JSMN_ARRAY || tokens[i].type == JSMN_OBJECT) {
						if (tokens[i].start != -1 && tokens[i].end == -1) {
							parser->toksuper = i;
							break;
						}
					}
				}
#endif
			}
			break;
#ifdef JSMN_STRICT
		/* In strict mode primitives are: numbers and booleans */
		case '-': case '0': case '1': case '2': case '3': case '4':
		case '5': case '6': case '7': case '8': case '9':
		case 't': case 'f': case 'n':
#else
		/* In non-strict mode every unquoted value is a primitive */
		default:
#endif
			r = jsmn_parse_primitive(parser, js, len, tokens, num_tokens);
			if (r < 0) {
				return r;
			}
			count++;
			if (parser->toksuper != -1 && tokens != NULL) {
				tokens[parser->toksuper].size++;
			}
			break;

#ifdef JSMN_STRICT
		/* Unexpected char in strict mode */
		default:
			return JSMN_ERROR_INVAL;
#endif
		}
	}

	if (tokens != NULL) {
		for (i = parser->toknext - 1; i >= 0; i--) {
			/* Unmatched opened object or array */
			if (tokens[i].start != -1 && tokens[i].end == -1) {
				return JSMN_ERROR_PART;
			}
		}
	}

	return count;
}

/**
 * Creates a new parser based over a given buffer with an array of tokens
 * available.
 */
void jsmn_init(jsmn_parser *parser) {
	parser->pos = 0;
	parser->toknext = 0;
	parser->toksuper = -1;
}

