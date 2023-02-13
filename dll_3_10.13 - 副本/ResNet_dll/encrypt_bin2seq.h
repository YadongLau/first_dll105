
// seqº”√‹
void Makecode(char* pstr, long length) {

	for (long i = 0; i < length; i++)
	{
		//*(pstr + i) = *(pstr + i) ^ pkey;

		if ((*(pstr + i)) <= 'z' && (*(pstr + i)) >= 'a')
		{
			if ((*(pstr + i)) == 'a')
			{
				(*(pstr + i)) = 'z';
			}
			else
			{
				(*(pstr + i)) = (char)((int)(*(pstr + i)) - 1);
			}

		}
		else if ((*(pstr + i)) <= 'Z' && (*(pstr + i)) >= 'A')
		{
			if ((*(pstr + i)) == 'Z')
			{
				(*(pstr + i)) = 'A';
			}
			else
			{
				(*(pstr + i)) = (char)((int)(*(pstr + i)) + 1);
			}

		}
		else if ((*(pstr + i)) <= '9' && (*(pstr + i)) >= '0')
		{
			if ((*(pstr + i)) == '0')
			{
				(*(pstr + i)) = '2';
			}
			else if ((*(pstr + i)) == '1')
			{
				(*(pstr + i)) = '5';
			}
			else if ((*(pstr + i)) == '2')
			{
				(*(pstr + i)) = '0';
			}
			else if ((*(pstr + i)) == '3')
			{
				(*(pstr + i)) = '7';
			}
			else if ((*(pstr + i)) == '4')
			{
				(*(pstr + i)) = '6';
			}
			else if ((*(pstr + i)) == '5')
			{
				(*(pstr + i)) = '1';
			}
			else if ((*(pstr + i)) == '6')
			{
				(*(pstr + i)) = '4';
			}
			else if ((*(pstr + i)) == '7')
			{
				(*(pstr + i)) = '3';
			}
			else if ((*(pstr + i)) == '8')
			{
				(*(pstr + i)) = '9';
			}
			else if ((*(pstr + i)) == '9')
			{
				(*(pstr + i)) = '8';
			}
		}
		else
		{
			(*(pstr + i)) = (*(pstr + i));
		}
	}
}

// seqΩ‚√‹
void Cutecode(char* pstr, long length)
{

	for (long i = 0; i < length; i++)
	{
		//*(pstr + i) = *(pstr + i) ^ pkey;

		if ((*(pstr + i)) <= 'z' && (*(pstr + i)) >= 'a')
		{
			if ((*(pstr + i)) == 'z')
			{
				(*(pstr + i)) = 'a';
			}
			else
			{
				(*(pstr + i)) = (char)((int)(*(pstr + i)) + 1);
			}

		}
		else if ((*(pstr + i)) <= 'Z' && (*(pstr + i)) >= 'A')
		{
			if ((*(pstr + i)) == 'A')
			{
				(*(pstr + i)) = 'Z';
			}
			else
			{
				(*(pstr + i)) = (char)((int)(*(pstr + i)) - 1);
			}
		}
		else if ((*(pstr + i)) <= '9' && (*(pstr + i)) >= '0')
		{
			if ((*(pstr + i)) == '0')
			{
				(*(pstr + i)) = '2';
			}
			else if ((*(pstr + i)) == '1')
			{
				(*(pstr + i)) = '5';
			}
			else if ((*(pstr + i)) == '2')
			{
				(*(pstr + i)) = '0';
			}
			else if ((*(pstr + i)) == '3')
			{
				(*(pstr + i)) = '7';
			}
			else if ((*(pstr + i)) == '4')
			{
				(*(pstr + i)) = '6';
			}
			else if ((*(pstr + i)) == '5')
			{
				(*(pstr + i)) = '1';
			}
			else if ((*(pstr + i)) == '6')
			{
				(*(pstr + i)) = '4';
			}
			else if ((*(pstr + i)) == '7')
			{
				(*(pstr + i)) = '3';
			}
			else if ((*(pstr + i)) == '8')
			{
				(*(pstr + i)) = '9';
			}
			else if ((*(pstr + i)) == '9')
			{
				(*(pstr + i)) = '8';
			}
		}
		else
		{
			(*(pstr + i)) = (*(pstr + i));
		}

	}

}

